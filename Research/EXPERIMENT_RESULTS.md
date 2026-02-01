# Comprehensive Model Strategy Experiments

**Date**: 2026-01-31
**Baseline**: LB 0.007809 (Sub 133 blend)
**Target**: LB 0.007

---

## Summary Table

| # | Experiment | Status | Best LOPO CV | Within-Player CV | Notes |
|---|------------|--------|--------------|------------------|-------|
| 1.1 | Stability-Adjusted Feature Selection | COMPLETE | 0.0319 | 0.0107 | 30 features, alpha=200 |
| 1.2 | Multi-Task Learning | COMPLETE | 0.0227 | 0.0126 | l1_ratio=0.1, alpha=1.0 |
| 1.3 | Feature Subspace Ensemble | COMPLETE | - | - | All 4 subsets FAILED profile |
| 2.1 | Gaussian Processes | COMPLETE | 0.0229 | - | RBF kernel, alpha=0.1 |
| 2.2 | Polynomial Interactions | COMPLETE | 0.0260 | - | 15 features, alpha=500, +2.9% improvement |
| 2.3 | LSTM Sequence Modeling | COMPLETE | 0.0395 | 0.0182 | GRU hidden=32, dropout=0.5 |
| 2.4 | Contrastive Learning | COMPLETE | 0.0347 | - | embedding_dim=64, -6% avg improvement |
| 2.5 | Fourier Phase Features | COMPLETE | 0.0260 | - | alpha=500 |
| 3.1 | Physics-Constrained NN | COMPLETE | 0.0458 | - | physics_weight=0.05 |
| 3.2 | Meta-Learning MAML | COMPLETE | 0.0271 | - | epochs=50, inner_lr=0.05, steps=5, +24% improvement |

---

## Detailed Results

### 1.1 Stability-Adjusted Feature Selection

**Script**: `scripts/stability_feature_selection.py`
**Status**: COMPLETE

**Objective**: Use only features with high importance + low drift for better generalization.

**Configuration**:
- Features: Top N from `stability_adjusted` score in `output/feature_drift_f4_with_importance.csv`
- Model: Ridge regression per player
- CV: LOPO and within-player 5-fold

**Results Grid**:

| n_features | alpha | LOPO MSE | Within-Player CV |
|------------|-------|----------|------------------|
| 20 | 10 | 0.0535 | 0.0132 |
| 20 | 200 | 0.0349 | 0.0109 |
| 30 | 200 | **0.0319** | **0.0107** |
| 40 | 200 | 0.0334 | 0.0107 |
| 60 | 200 | 0.0344 | 0.0108 |
| 100 | 200 | 0.0390 | 0.0117 |

**Best Configuration**:
- n_features: 30
- alpha: 200
- LOPO MSE: 0.0319
- Within-player CV: 0.0107
- Per-target: angle=0.0565, depth=0.0214, left_right=0.0178

**Key Findings**:
1. LOPO CV (0.032) is much higher than within-player CV (0.011) and LB scores (0.008)
2. This confirms: **LOPO is NOT representative of LB performance**
3. Within-player CV (0.0107) is closer to our best LB (0.0078)
4. Sweet spot is around 30 stable features - more features hurt LOPO

**Interpretation**:
- The stability-adjusted feature selection helps reduce overfitting
- However, LOPO is fundamentally different from the test distribution
- The test set likely has similar within-player distribution, not leave-one-player-out

---

### 1.2 Multi-Task Learning

**Script**: `scripts/multitask_model.py`
**Status**: COMPLETE

**Objective**: Train single model predicting all 3 targets jointly using MultiTaskElasticNet.

**Results Grid** (best configs):

| l1_ratio | alpha | Multitask LOPO | Separate LOPO | Within-Player CV |
|----------|-------|----------------|---------------|------------------|
| 0.1 | 1.0 | **0.0230** | 0.0230 | 0.0126 |
| 0.3 | 0.5 | 0.0227 | 0.0227 | 0.0126 |
| 0.5 | 0.5 | 0.0227 | 0.0227 | 0.0126 |

**Key Findings**:
1. Multitask and Separate models converge to similar performance with high regularization
2. Best LOPO: 0.0227 (l1_ratio=0.3-0.9, alpha>=0.5)
3. Within-player CV: 0.0126 (consistent with other approaches)
4. **Multitask does NOT provide significant benefit over separate models**

**Interpretation**:
- The hypothesis that shared representations help was not confirmed
- High regularization dominates - both approaches reduce to similar solutions
- No evidence that targets share useful structure for joint learning

---

### 1.3 Feature Subspace Ensemble

**Script**: `scripts/subspace_ensemble.py`
**Status**: COMPLETE

**Objective**: Create models with non-overlapping feature subsets, blend only those passing profile check.

**Configuration**:
- 4 subsets of 15 features each (non-overlapping from top 60 stable features)
- Profile check: angle_std < 0.145, depth_mean in [0.49, 0.52]

**Results**:

| Subset | CV MSE | angle_std | depth_mean | Profile |
|--------|--------|-----------|------------|---------|
| 1 | 0.0380 | 4.24 | 9.63 | FAIL |
| 2 | 0.0229 | 6.01 | 7.10 | FAIL |
| 3 | 0.0281 | 4.36 | 10.32 | FAIL |
| 4 | 0.0292 | 4.55 | 9.07 | FAIL |

**Key Findings**:
1. **All 4 subsets failed profile check** - angle_std and depth_mean way outside acceptable range
2. No ensemble was created
3. Individual Ridge models on feature subsets do not produce submission-quality predictions

**Interpretation**:
- Feature subsets alone are insufficient for good predictions
- Profile failure suggests these models are not learning the correct distribution
- Need full feature set or different approach

---

### 2.1 Gaussian Processes with ARD Kernel

**Script**: `scripts/gp_model.py`
**Status**: COMPLETE

**Objective**: GPs provide uncertainty estimates + automatic feature relevance.

**Configuration**:
- Kernels: RBF, Matern
- Alpha (noise): 0.001, 0.01, 0.1
- Features: Top 20 stable features

**Results**:

| kernel | alpha | LOPO MSE | Calibration (angle) | Calibration (depth) |
|--------|-------|----------|---------------------|---------------------|
| rbf | 0.001 | 0.0275 | 0.40 | 0.50 |
| rbf | 0.01 | 0.0275 | 0.40 | 0.51 |
| **rbf** | **0.1** | **0.0229** | 0.22 | 0.51 |
| matern | 0.01 | 0.0277 | 0.31 | 0.75 |
| matern | 0.1 | 0.0242 | 0.21 | 0.75 |

**Key Findings**:
1. Best LOPO: 0.0229 with RBF kernel, alpha=0.1
2. **Uncertainty is POORLY calibrated** - negative correlation with actual error for angle/depth
3. Low-uncertainty predictions actually have HIGHER error (ratio < 1)
4. Only left_right shows reasonable calibration (ratio ~1.1)

**Interpretation**:
- GP achieves competitive LOPO MSE (similar to multitask)
- However, uncertainty estimates are unreliable and cannot be used for prediction weighting
- The ARD feature relevance could still be useful for feature selection

---

### 2.2 Polynomial Interaction Features

**Script**: `scripts/polynomial_model.py`
**Status**: COMPLETE

**Objective**: Degree-2 polynomial on top stable features with heavy regularization.

**Results Grid**:

| n_base | alpha | Baseline | Poly2 | Improvement |
|--------|-------|----------|-------|-------------|
| 10 | 500 | 0.0247 | 0.0279 | -12.8% |
| 15 | 500 | 0.0268 | **0.0260** | **+2.9%** |
| 20 | 500 | 0.0292 | 0.0266 | +8.7% |
| 25 | 500 | 0.0280 | 0.0414 | -47.8% |
| 30 | 500 | 0.0274 | 0.0361 | -31.7% |

**Best Configuration**:
- n_base_features: 15
- alpha: 500
- Baseline LOPO: 0.0268
- Poly2 LOPO: 0.0260
- Improvement: +2.9%

**Submission Profile**: FAIL (angle_std=5.55, depth_mean=9.66)

**Key Findings**:
1. Polynomial features provide marginal improvement (+2.9%) only with 15-20 base features
2. More features cause polynomial explosion and severe overfitting
3. **Profile check failed** - predictions way outside acceptable range
4. Heavy regularization (alpha=500) required but still insufficient

**Interpretation**:
- Polynomial interactions add complexity without improving generalization significantly
- The degree-2 expansion creates too many features relative to sample size
- Not a viable approach for this problem

---

### 2.3 LSTM Sequence Modeling

**Script**: `scripts/lstm_model.py`
**Status**: COMPLETE

**Objective**: Capture temporal patterns with recurrent networks.

**Configuration**:
- Models: LSTM, GRU
- Hidden dims: 16, 32, 64
- Dropout: 0.3, 0.5
- Sequence: 60 frames (downsampled from 240)

**Results**:

| Model | Hidden | Dropout | LOPO MSE | Within-Player MSE |
|-------|--------|---------|----------|-------------------|
| LSTM | 16 | 0.3 | 0.0713 | 0.0170 |
| LSTM | 32 | 0.5 | 0.0755 | 0.0164 |
| LSTM | 64 | 0.5 | 0.1171 | 0.0160 |
| GRU | 16 | 0.3 | 0.0503 | 0.0173 |
| **GRU** | **32** | **0.5** | **0.0395** | 0.0182 |
| GRU | 64 | 0.5 | 0.0614 | 0.0173 |

**Key Findings**:
1. Best LOPO: 0.0395 with GRU hidden=32, dropout=0.5
2. **LSTM/GRU perform significantly worse than linear models** on LOPO
3. Within-player CV (~0.017) is similar across all configs
4. Larger models overfit more (LSTM-64: 0.117 LOPO)
5. GRU consistently outperforms LSTM

**Interpretation**:
- Recurrent models capture within-player patterns well but fail to generalize across players
- The temporal patterns learned are player-specific, not universal
- Linear models with good features outperform sequence models on this task

---

### 2.4 Contrastive Learning

**Script**: `scripts/contrastive_learning.py`
**Status**: COMPLETE

**Objective**: Pre-train encoder to embed similar-outcome shots nearby regardless of player.

**Configuration**:
- Triplet loss with margin=1.0
- Embedding dims: 16, 32, 64
- 50 epochs training

**Results** (averaged across 5 LOPO folds):

| embedding_dim | Contrastive MSE | Baseline MSE | Improvement |
|---------------|-----------------|--------------|-------------|
| 16 | 0.0436 | 0.0351 | -41.1% |
| 32 | 0.0536 | 0.0351 | -69.0% |
| **64** | **0.0347** | 0.0351 | **-6.4%** |

**Per-Player Results (embedding_dim=64)**:

| Test Player | Contrastive | Baseline | Improvement |
|-------------|-------------|----------|-------------|
| 1 | 0.0496 | 0.0493 | -0.6% |
| 2 | 0.0388 | 0.0241 | -61.3% |
| 3 | 0.0128 | 0.0117 | -9.7% |
| 4 | 0.0301 | 0.0374 | +19.5% |
| 5 | 0.0423 | 0.0531 | +20.3% |

**Key Findings**:
1. **Contrastive learning hurts generalization on average**
2. Performance is highly variable across players (-61% to +20%)
3. The encoder learns player-specific features rather than player-invariant ones
4. Larger embedding dims reduce the damage but still negative

**Interpretation**:
- The hypothesis that similar outcomes cluster regardless of player was wrong
- Player mechanics are too distinct - same outcome can have very different motion patterns
- Contrastive learning is not suitable for this cross-player generalization task

---

### 2.5 Fourier Phase Features

**Script**: `scripts/fourier_phase.py`
**Status**: COMPLETE

**Objective**: Add phase at dominant frequency and cross-keypoint phase differences.

**Results**:

| alpha | LOPO MSE | angle | depth | left_right |
|-------|----------|-------|-------|------------|
| 10 | 0.0344 | 0.0735 | 0.0156 | 0.0142 |
| 50 | 0.0325 | 0.0677 | 0.0156 | 0.0142 |
| 100 | 0.0308 | 0.0625 | 0.0157 | 0.0142 |
| 200 | 0.0286 | 0.0559 | 0.0158 | 0.0142 |
| **500** | **0.0260** | 0.0475 | 0.0162 | 0.0142 |

**Key Findings**:
1. Best LOPO: 0.0260 with alpha=500
2. Phase features help primarily for **angle** prediction (0.047 vs 0.057 for stability features)
3. Depth and left_right predictions are similar to other approaches
4. High regularization (alpha=500) required - suggests noise in phase features

**Interpretation**:
- Fourier phase features contain useful timing information for angle prediction
- The kinetic chain timing (phase differences) may capture release mechanics
- However, features require heavy regularization to generalize

---

### 3.1 Physics-Constrained Neural Network

**Script**: `scripts/physics_constrained_nn.py`
**Status**: COMPLETE

**Objective**: Network predicts release parameters. Loss includes physics consistency.

**Configuration**:
- Physics loss weight: 0.0, 0.01, 0.05, 0.1, 0.2
- 2-layer MLP with dropout

**Results**:

| physics_weight | LOPO MSE | angle | depth | left_right |
|----------------|----------|-------|-------|------------|
| 0.0 | 0.0518 | 0.0631 | 0.0553 | 0.0370 |
| 0.01 | 0.0703 | 0.1052 | 0.0685 | 0.0373 |
| **0.05** | **0.0458** | 0.0607 | 0.0474 | 0.0292 |
| 0.1 | 0.0610 | 0.0815 | 0.0541 | 0.0474 |
| 0.2 | 0.0644 | 0.1008 | 0.0425 | 0.0497 |

**Key Findings**:
1. Best LOPO: 0.0458 with physics_weight=0.05
2. **Physics constraint does not help** - performance worse than simple linear models
3. Too little physics weight (0.01) hurts angle; too much (0.2) hurts overall
4. Neural network alone (weight=0) already underperforms linear models

**Interpretation**:
- The physics equations may not accurately model the release-to-outcome relationship
- Or the network cannot learn to satisfy physics constraints while also fitting the data
- Simple linear models remain superior for this task

---

### 3.2 Meta-Learning MAML

**Script**: `scripts/maml_model.py`
**Status**: COMPLETE

**Objective**: MAML finds initialization that adapts quickly to new player with few samples.

**Configuration**:
- Meta epochs: 50, 100
- Inner LR: 0.01, 0.05
- Inner steps: 3, 5

**Results**:

| epochs | inner_lr | steps | MAML MSE | Baseline | Improvement |
|--------|----------|-------|----------|----------|-------------|
| 50 | 0.01 | 3 | 0.0361 | 0.0355 | -1.4% |
| 50 | 0.01 | 5 | 0.0324 | 0.0355 | +8.8% |
| 50 | 0.05 | 3 | 0.0376 | 0.0355 | -5.8% |
| **50** | **0.05** | **5** | **0.0271** | 0.0355 | **+23.9%** |
| 100 | 0.01 | 3 | 0.0369 | 0.0355 | -3.8% |
| 100 | 0.01 | 5 | 0.0336 | 0.0355 | +5.4% |
| 100 | 0.05 | 3 | 0.0299 | 0.0355 | +16.0% |
| 100 | 0.05 | 5 | 0.0309 | 0.0355 | +13.1% |

**Best Configuration**:
- Meta epochs: 50
- Inner LR: 0.05
- Inner steps: 5
- MAML MSE: 0.0271
- Improvement: +23.9% over Ridge baseline

**Key Findings**:
1. MAML achieves best improvement over baseline (+24%) among all experiments
2. Optimal config: fewer meta-epochs (50), higher inner LR (0.05), more steps (5)
3. More meta-epochs (100) can overfit the meta-learning process
4. MAML captures some shared structure that helps adaptation

**Interpretation**:
- MAML successfully learns an initialization that transfers across players
- The shared structure is subtle - requires fast adaptation (high inner LR, many steps)
- However, LOPO MSE (0.027) is still worse than GP (0.023) or multitask (0.023)

---

## Overall Rankings (by LOPO MSE)

| Rank | Experiment | Best LOPO MSE | Notes |
|------|------------|---------------|-------|
| 1 | Multi-Task Learning (1.2) | 0.0227 | High regularization dominates |
| 2 | Gaussian Processes (2.1) | 0.0229 | RBF kernel, alpha=0.1 |
| 3 | Fourier Phase (2.5) | 0.0260 | Phase helps angle prediction |
| 4 | Polynomial (2.2) | 0.0260 | Marginal +2.9% improvement |
| 5 | MAML (3.2) | 0.0271 | +24% vs baseline |
| 6 | Stability Features (1.1) | 0.0319 | 30 features optimal |
| 7 | Contrastive (2.4) | 0.0347 | Hurts generalization |
| 8 | LSTM/GRU (2.3) | 0.0395 | Poor cross-player transfer |
| 9 | Physics NN (3.1) | 0.0458 | Constraint doesn't help |
| 10 | Subspace Ensemble (1.3) | - | All failed profile |

---

## Critical Insight: CV-LB Gap

From experiment 1.1, we observe:
- LOPO CV: 0.032
- Within-player CV: 0.011
- Best LB: 0.008

This suggests:
1. **Within-player CV is more representative than LOPO**
2. The test set likely samples from the same players as training
3. Models should be optimized for within-player generalization, not cross-player

---

## Conclusions

1. **Linear models with regularization remain the best approach** - Multitask/GP achieve best LOPO
2. **Neural approaches (LSTM, Physics NN, Contrastive) underperform** - not enough data for complex models
3. **MAML shows promise** for transfer learning but still worse than simple approaches
4. **Profile validation is critical** - many models fail to produce submission-quality predictions
5. **LOPO CV does not predict LB** - within-player metrics are more relevant

## Next Steps

1. Focus on improving within-player CV while maintaining good profile
2. Consider ensemble of top 3 approaches (Multitask, GP, Fourier) if they pass profile check
3. The path to LB 0.007 may require finding features that generalize within-player better
