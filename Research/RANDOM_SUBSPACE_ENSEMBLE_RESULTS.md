# Random Subspace Ensemble Results

**Date**: 2026-02-12
**Script**: scripts/random_subspace_ensemble.py
**Conclusion**: DOES NOT HELP - ensemble LOO MSE is 3.6% worse than baseline

## Approach

Train multiple per-example locally weighted Ridge models on random feature subsets
and average predictions. Classic random subspace method for variance reduction.

Features: 223 per target (198 hoop-relative + 15 PLS + 10 joint angles).

## Methodology

1. **Subset size selection**: Tested frac=0.50, 0.60, 0.70, 0.80 with 10 subsets each
2. **Stratified vs random**: Compared proportional group sampling vs fully random
3. **Optimal N**: Tested N=5, 10, 20, 30 subsets
4. **Stability**: 3 seeds (42, 123, 999) per config

## Results

### Phase 2: Baseline (full features, locally weighted Ridge)
| Target | LOO MSE |
|--------|---------|
| angle | 0.002687 |
| depth | 0.004613 |
| left_right | 0.004328 |
| **mean** | **0.003876** |

### Phase 3: Subset Size Selection (10 subsets, seed=42)

**Angle:**
| frac | Ensemble MSE | Individual MSE (mean +/- std) | Pairwise Corr |
|------|-------------|-------------------------------|----------------|
| 0.50 | 0.003160 | 0.003663 +/- 0.000714 | 0.9749 |
| 0.60 | 0.003092 | 0.003510 +/- 0.000736 | 0.9792 |
| 0.70 | 0.003114 | 0.003458 +/- 0.000606 | 0.9829 |
| **0.80** | **0.002802** | 0.002990 +/- 0.000418 | 0.9907 |

**Depth:**
| frac | Ensemble MSE | Individual MSE | Pairwise Corr |
|------|-------------|----------------|----------------|
| 0.50 | 0.004845 | 0.005247 +/- 0.000573 | 0.9631 |
| 0.60 | 0.004873 | 0.005213 +/- 0.000556 | 0.9693 |
| 0.70 | 0.004845 | 0.005114 +/- 0.000588 | 0.9760 |
| **0.80** | **0.004570** | 0.004714 +/- 0.000467 | 0.9874 |

**Left_right:**
| frac | Ensemble MSE | Individual MSE | Pairwise Corr |
|------|-------------|----------------|----------------|
| 0.50 | 0.004790 | 0.005346 +/- 0.000721 | 0.9288 |
| 0.60 | 0.004684 | 0.005125 +/- 0.000756 | 0.9454 |
| 0.70 | 0.004668 | 0.005018 +/- 0.000730 | 0.9574 |
| **0.80** | **0.004426** | 0.004623 +/- 0.000409 | 0.9767 |

Best frac=0.80 across all targets (minimizes individual MSE degradation).

### Phase 4: Stratified vs Random

| Target | Random MSE | Stratified MSE | Winner |
|--------|-----------|----------------|--------|
| angle | 0.002802 | 0.003254 | random |
| depth | 0.004570 | 0.004867 | random |
| left_right | 0.004426 | 0.004708 | random |

Fully random consistently better than stratified.

### Phase 5: Optimal N

| Target | N=5 | N=10 | N=20 | N=30 | Best N |
|--------|-----|------|------|------|--------|
| angle | 0.002891 | **0.002802** | 0.003011 | 0.002976 | 10 |
| depth | **0.004496** | 0.004570 | 0.004726 | 0.004697 | 5 |
| left_right | 0.004574 | **0.004426** | 0.004485 | 0.004506 | 10 |

More subsets beyond 10 does NOT help - diminishing returns with seed=42. With different
seeds N=5 was sometimes best. This suggests N=10 captures most of the variance reduction.

### Phase 6: Stability (3 seeds, 3-seed average)

| Target | Seed 42 | Seed 123 | Seed 999 | 3-seed avg | Baseline | Delta |
|--------|---------|----------|----------|------------|----------|-------|
| angle | 0.002802 | 0.003046 | 0.002977 | 0.002930 | 0.002687 | +9.0% |
| depth | 0.004496 | 0.004782 | 0.004733 | 0.004643 | 0.004613 | +0.7% |
| left_right | 0.004426 | 0.004514 | 0.004527 | 0.004476 | 0.004328 | +3.4% |
| **mean** | | | | **0.004016** | **0.003876** | **+3.6%** |

Prediction std across seeds:
- angle: test=0.002419, oof=0.002672
- depth: test=0.004824, oof=0.004190
- left_right: test=0.002816, oof=0.003094

## Diversity Analysis

Correlation with Sub 784:
- angle: r=0.9416
- depth: r=0.9471
- left_right: r=0.8773

Correlation with Sub 2063 (current best):
- angle: r=0.9835
- depth: r=0.9774
- left_right: r=0.9840

Very high correlation with best submission - no useful diversity.

## Generated Submissions

| Sub | Description |
|-----|-------------|
| 2156 | STANDALONE random subspace ensemble |
| 2157 | Blend with Sub 784 (aw=0.00 dw=0.30 lw=0.50) |
| 2158 | Blend with Sub 784 (aw=0.00 dw=0.20 lw=0.30) |
| 2159 | 10% ensemble + 90% Sub 2063 |
| 2160 | 20% ensemble + 80% Sub 2063 |
| 2161 | 30% ensemble + 70% Sub 2063 |
| 2162 | 50% ensemble + 50% Sub 2063 |

## Analysis: Why It Failed

1. **Pairwise correlation too high**: At frac=0.80 (best individual MSE), pairwise
   correlation is 0.97-0.99. With such high correlation, averaging N models only
   reduces variance by factor ~1/(1 + (N-1)*rho) which is negligible when rho=0.99.

2. **Bias-variance tradeoff**: Lower frac (0.50) creates more diversity (corr=0.93)
   but each subset model is ~20-40% worse. The increased bias dominates the variance
   reduction benefit.

3. **Feature redundancy**: The 223 features are mostly hoop-relative coordinates of
   12 joints at one frame plus summary stats. Missing 20% of features barely changes
   the model because nearby features are highly correlated (e.g., right_wrist_x position
   at frame 153 correlates with right_elbow_x).

4. **LOO is optimistic**: The baseline LOO MSE (0.003876) is already lower than the
   actual LB MSE (~0.006619). Random subspace makes LOO even more optimistic by averaging
   over many LOO estimates, but this doesn't transfer to test.

5. **Stability issue**: Even with 10 subsets, predictions vary by ~0.003 across seeds,
   indicating the method is noisy - not a reliable variance reduction technique for
   this problem size (345 train, 113 test).

## Key Takeaway

Random subspace ensemble is theoretically sound for variance reduction but fails here
because:
- Features are too correlated to produce diverse subsets
- 345 training samples is too small - each subset model is unstable
- The locally weighted Ridge is already effectively regularized; removing features adds
  noise without improving generalization

This approach is a **dead end** for this competition.
