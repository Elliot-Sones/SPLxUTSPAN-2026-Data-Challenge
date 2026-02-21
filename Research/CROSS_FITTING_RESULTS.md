# Cross-Fitting Variance Reduction Results

Date: 2026-02-12

Script: scripts/cross_fitting.py

## Concept
Split training data into K folds per-player. For each fold, train per-example Ridge
on (K-1) folds only. Average K predictions for test. Benefits: variance reduction,
reduced overfitting to specific noisy samples.

## Configuration
- Per-example locally weighted Ridge (Gaussian kernel, alpha=10)
- bandwidth_quantile=0.3
- Features: 198 HC + 15 PLS per target at target-specific frames (angle=153, depth=150, LR=170)
- Per-player splitting (each player's data split separately)
- 345 train shots, 113 test shots, 5 players (~66-74 shots each)

## ANGLE

### Phase 1: Baseline
- LOO MSE: 0.002645

### Phase 2: K-Fold Cross-Fitting
| K | LOO MSE | Delta | Test Std | Fold Corr |
|---|---------|-------|----------|-----------|
| 3 | 0.003689 | +0.001044 | 0.016618 | 0.9751 |
| 5 | 0.003320 | +0.000675 | 0.013868 | 0.9868 |
| 7 | 0.003105 | +0.000460 | 0.011662 | 0.9911 |
| 10 | 0.002944 | +0.000299 | 0.009986 | 0.9942 |

### Phase 3: Weighted Averaging (inverse-MSE)
- K=3: weight range [0.2547, 0.3791], diff from equal: 6.18e-06
- K=5: weight range [0.1297, 0.3428], diff from equal: 5.87e-06
- K=7: weight range [0.1047, 0.2407], diff from equal: 2.38e-06
- K=10: weight range [0.0701, 0.2048], diff from equal: 1.45e-06

### Phase 4: Bootstrap Bagging
- B=10: test_std=0.027081, bag_corr=0.9572
- B=20: test_std=0.027749, bag_corr=0.9607

### Phase 5: Stability (K=5, 5 seeds)
- LOO MSE mean: 0.003188 +/- 0.000063
- Test pred std across seeds: 0.001762

### Phase 6: Effective Sample Size
| K | Eff Samples | Full Samples | Fold Corr | Theoretical Var Frac |
|---|-------------|--------------|-----------|---------------------|
| 3 | 46.4 | 69.0 | 0.9751 | 0.9834 |
| 5 | 55.6 | 69.0 | 0.9868 | 0.9894 |
| 7 | 59.6 | 69.0 | 0.9911 | 0.9924 |
| 10 | 62.6 | 69.0 | 0.9942 | 0.9948 |

### Phase 7: Diversity (correlation with Sub 2063 / Sub 784)
- Baseline: r_2063=0.9806, r_784=0.9362
- K=3: r_2063=0.9893, r_784=0.9535
- K=5: r_2063=0.9855, r_784=0.9451
- K=10: r_2063=0.9829, r_784=0.9402
- Bag B=20: r_2063=0.9899, r_784=0.9568
- K=5 multiseed: r_2063=0.9856, r_784=0.9453

## DEPTH

### Phase 1: Baseline
- LOO MSE: 0.004601

### Phase 2: K-Fold Cross-Fitting
| K | LOO MSE | Delta | Test Std | Fold Corr |
|---|---------|-------|----------|-----------|
| 3 | 0.005425 | +0.000823 | 0.025034 | 0.8532 |
| 5 | 0.005758 | +0.001156 | 0.020392 | 0.9198 |
| 7 | 0.005217 | +0.000616 | 0.017284 | 0.9566 |
| 10 | 0.004861 | +0.000259 | 0.013703 | 0.9752 |

### Phase 3: Weighted Averaging (inverse-MSE)
- K=3: weight range [0.2712, 0.4216], diff from equal: 1.361e-05
- K=5: weight range [0.1371, 0.2901], diff from equal: 8.40e-06
- K=7: weight range [0.0930, 0.2211], diff from equal: 7.17e-06
- K=10: weight range [0.0595, 0.1604], diff from equal: 3.68e-06

### Phase 4: Bootstrap Bagging
- B=10: test_std=0.031292, bag_corr=0.8637
- B=20: test_std=0.030923, bag_corr=0.8932

### Phase 5: Stability (K=5, 5 seeds)
- LOO MSE mean: 0.005068 +/- 0.000225
- Test pred std across seeds: 0.002165

### Phase 6: Effective Sample Size
| K | Eff Samples | Full Samples | Fold Corr | Theoretical Var Frac |
|---|-------------|--------------|-----------|---------------------|
| 3 | 46.4 | 69.0 | 0.8532 | 0.9021 |
| 5 | 55.6 | 69.0 | 0.9198 | 0.9359 |
| 7 | 59.6 | 69.0 | 0.9566 | 0.9628 |
| 10 | 62.6 | 69.0 | 0.9752 | 0.9777 |

### Phase 7: Diversity (correlation with Sub 2063 / Sub 784)
- Baseline: r_2063=0.9734, r_784=0.9400
- K=3: r_2063=0.9760, r_784=0.9474
- K=5: r_2063=0.9769, r_784=0.9467
- K=10: r_2063=0.9761, r_784=0.9443
- Bag B=20: r_2063=0.9777, r_784=0.9511
- K=5 multiseed: r_2063=0.9779, r_784=0.9478

## LEFT_RIGHT

### Phase 1: Baseline
- LOO MSE: 0.004331

### Phase 2: K-Fold Cross-Fitting
| K | LOO MSE | Delta | Test Std | Fold Corr |
|---|---------|-------|----------|-----------|
| 3 | 0.005870 | +0.001539 | 0.019735 | 0.9136 |
| 5 | 0.004932 | +0.000602 | 0.015146 | 0.9645 |
| 7 | 0.004848 | +0.000517 | 0.013897 | 0.9733 |
| 10 | 0.004299 | -0.000032 | 0.010781 | 0.9851 |

### Phase 3: Weighted Averaging (inverse-MSE)
- K=3: weight range [0.2531, 0.4346], diff from equal: 1.266e-05
- K=5: weight range [0.0988, 0.3084], diff from equal: 9.46e-06
- K=7: weight range [0.0864, 0.2084], diff from equal: 1.87e-06
- K=10: weight range [0.0357, 0.1646], diff from equal: 3.24e-06

### Phase 4: Bootstrap Bagging
- B=10: test_std=0.030416, bag_corr=0.8753
- B=20: test_std=0.029899, bag_corr=0.8847

### Phase 5: Stability (K=5, 5 seeds)
- LOO MSE mean: 0.004693 +/- 0.000228
- Test pred std across seeds: 0.001886

### Phase 6: Effective Sample Size
| K | Eff Samples | Full Samples | Fold Corr | Theoretical Var Frac |
|---|-------------|--------------|-----------|---------------------|
| 3 | 46.4 | 69.0 | 0.9136 | 0.9424 |
| 5 | 55.6 | 69.0 | 0.9645 | 0.9716 |
| 7 | 59.6 | 69.0 | 0.9733 | 0.9771 |
| 10 | 62.6 | 69.0 | 0.9851 | 0.9866 |

### Phase 7: Diversity (correlation with Sub 2063 / Sub 784)
- Baseline: r_2063=0.9818, r_784=0.8648
- K=3: r_2063=0.9828, r_784=0.8840
- K=5: r_2063=0.9852, r_784=0.8783
- K=10: r_2063=0.9838, r_784=0.8713
- Bag B=20: r_2063=0.9860, r_784=0.8915
- K=5 multiseed: r_2063=0.9853, r_784=0.8777

## Submissions Generated

### Standalone
| Sub | Variant | angle_std | depth_mean |
|-----|---------|-----------|------------|
| 2202 | baseline | 0.159809 | 0.511734 |
| 2203 | cf_K5 | 0.157625 | 0.511839 |
| 2204 | cf_K5_weighted | 0.157875 | 0.511773 |
| 2205 | cf_K5_multiseed | 0.158084 | 0.511483 |
| 2206 | cf_K10 | 0.158879 | 0.511680 |
| 2207 | bag_B20 | 0.155515 | 0.510479 |

### Blended with Sub 784 (aw=0.50, dw=0.30, lw=0.50)
| Sub | Variant | angle_std | depth_mean |
|-----|---------|-----------|------------|
| 2208 | baseline + 784 | 0.150827 | 0.512900 |
| 2209 | cf_K5 + 784 | 0.150096 | 0.512931 |
| 2210 | cf_K5_weighted + 784 | 0.150166 | 0.512911 |
| 2211 | cf_K5_multiseed + 784 | 0.150327 | 0.512825 |
| 2212 | cf_K10 + 784 | 0.150523 | 0.512884 |
| 2213 | bag_B20 + 784 | 0.149499 | 0.512523 |

### Blended with Sub 2063
| Sub | Config | angle_std | depth_mean |
|-----|--------|-----------|------------|
| 2214 | 10% cf_K5 + 90% Sub2063 | 0.150829 | 0.512413 |
| 2215 | 20% cf_K5 + 80% Sub2063 | 0.151404 | 0.512349 |
| 2216 | 30% cf_K5 + 70% Sub2063 | 0.152026 | 0.512286 |
| 2217 | 10% cf_K5_multiseed + 90% Sub2063 | 0.150876 | 0.512378 |
| 2218 | 20% cf_K5_multiseed + 80% Sub2063 | 0.151497 | 0.512278 |
| 2219 | 30% cf_K5_multiseed + 70% Sub2063 | 0.152165 | 0.512179 |
| 2220 | 10% bag_B20 + 90% Sub2063 | 0.150682 | 0.512277 |
| 2221 | 20% bag_B20 + 80% Sub2063 | 0.151094 | 0.512077 |
| 2222 | 30% bag_B20 + 70% Sub2063 | 0.151538 | 0.511878 |

## Key Findings

### 1. Cross-fitting LOO is HIGHER than baseline (more honest)
- Baseline LOO uses ALL training data and leaves out only self-weight. Cross-fitting LOO
  holds out an entire fold, providing a less optimistic estimate.
- K=10 is closest to baseline LOO (smaller held-out fold), K=3 has highest LOO inflation.
- For LR at K=10, LOO is actually LOWER than baseline (-0.000032), suggesting the baseline
  LOO for LR may be slightly optimistic even in standard mode.

### 2. Fold correlations are extremely high (0.85-0.99)
- Angle: fold corr 0.975-0.994 (very high, limited diversity between folds)
- Depth: fold corr 0.853-0.975 (most diverse, especially at low K)
- LR: fold corr 0.914-0.985
- High fold correlation means the theoretical variance reduction is minimal:
  var_avg/var_single = (1/K)(1 + (K-1)*rho). With rho=0.97, K=5: 97.6% of original variance.
- The sub-models are too similar because the kernel-weighted Ridge is dominated by the
  closest neighbors, which appear in most folds.

### 3. Weighted averaging makes negligible difference
- diff from equal-weight averaging: 1e-06 to 1e-05 scale
- Not enough fold-to-fold MSE variation to meaningfully weight

### 4. Bootstrap bagging has HIGHER variance than cross-fitting
- Bag test_std (0.027-0.031) >> cross-fitting test_std (0.010-0.025)
- Bootstrap with replacement creates more diverse subsets, but also noisier
- Lower bag_corr (0.86-0.96) vs fold_corr, confirming more diversity but at cost of noise

### 5. Stability across seeds is reasonable
- LOO MSE std across 5 seeds: 0.000063 (angle), 0.000225 (depth), 0.000228 (LR)
- Test pred std across seeds: 0.0018-0.0022 (small relative to predictions)
- Multi-seed averaging provides marginal smoothing

### 6. Diversity with Sub 2063 is VERY HIGH (r>0.97)
- All variants correlate >0.97 with Sub 2063 for all targets
- This is expected since both use the same underlying approach (per-example Ridge)
- Cross-fitting does NOT produce meaningfully different predictions from baseline
- The predictions that differ most from Sub 2063 are for DEPTH baseline (r=0.9734)

### 7. Depth has most diversity potential
- Depth fold correlations are lowest (0.853 at K=3), meaning fold sub-models disagree most
- This is where cross-fitting could theoretically help most
- But even here, the theoretical variance reduction is only ~10% at K=3

### Bottom Line
Cross-fitting produces marginally different predictions that are extremely correlated with
the baseline. The variance reduction is theoretically limited by high fold correlations
(rho > 0.85). The approach is unlikely to produce meaningful LB improvement as a standalone
method, but could serve as a mild regularizer when blended at small weights with Sub 2063.

Best candidates for LB testing:
1. Sub 2214 (10% cf_K5 + 90% Sub2063) - minimal perturbation, slight regularization
2. Sub 2209 (cf_K5 + 784 blend) - standard blend format for comparison
3. Sub 2220 (10% bag_B20 + 90% Sub2063) - bootstrap variant as alternative
