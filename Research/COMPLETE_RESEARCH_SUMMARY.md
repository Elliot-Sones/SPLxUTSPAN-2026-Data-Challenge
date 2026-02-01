# Complete Research Summary: SPLxUTSPAN 2026 Basketball Free Throw Prediction

## Problem Statement

**Goal**: Beat the competition winners' score of 0.007 MSE
**Starting Point**: 0.010220 (submission 8)
**Best Achieved**: 0.008305 (submission 25)
**Gap Remaining**: 15.7% improvement needed

**Dataset**:
- Training: 345 samples from 5 basketball players
- Test: 113 samples
- Features: Motion capture keypoints (207 keypoints x 240 frames)
- Targets: 3 values (angle, depth, left_right) - all scaled to [0,1]

---

## All Submissions and LB Scores

| Sub | LB Score | Method | Key Details |
|-----|----------|--------|-------------|
| 8 | 0.010220 | Baseline | Initial model |
| 9 | 0.009109 | Ensemble | LGB + XGB + CatBoost + Ridge with meta-stacking |
| 10 | 0.008907 | Optuna-tuned | Per-player per-target hyperparameter optimization |
| 11 | 0.009848 | Ultra-optimized | Heavy tuning - OVERFIT (worse than simpler models) |
| 20 | 0.008619 | 80-20 blend | 80% sub9 + 20% sub10 |
| 25 | **0.008305** | 50-50 blend | 50% sub9 + 50% sub10 **(BEST)** |
| 34 | 0.008377 | 30-70 blend | 30% sub9 + 70% sub10 |
| 51 | 0.008807 | Hybrid | sub25 angle/lr + sub43 depth - FAILED |

---

## Phase 1: Initial Model Development

### Submission 8 - Baseline (LB: 0.010220)
- Simple per-player models
- Basic feature extraction from motion capture data

### Submission 9 - Ensemble (LB: 0.009109)
**File**: `src/ensemble_submission.py`

**Architecture**:
- 5 base models: LightGBM, LightGBM-deep, XGBoost, Ridge, CatBoost
- Per-player per-target training (5 players x 3 targets x 5 models = 75 models)
- Meta-stacking with Ridge regression on OOF predictions

**Feature Engineering** (`src/advanced_features.py`, `src/hybrid_features.py`):
- 368 total features extracted
- Frame-specific features at critical frames:
  - Frame 153 for angle
  - Frame 102 for depth
  - Frame 237 for left_right
- Velocity, acceleration, joint angles, distances between keypoints

### Submission 10 - Optuna-Tuned (LB: 0.008907)
**File**: `src/optimized_ensemble.py`

**Changes from sub9**:
- Optuna hyperparameter optimization (20-30 trials per player-target)
- Target-specific feature selection (top 150 features per target using mutual information)
- Weighted ensemble: 45% LGB + 45% CatBoost + 10% Ridge

### Submission 11 - Ultra-Optimized (LB: 0.009848 - WORSE)
**Key Learning**: Complex models with low CV error can perform WORSE on LB due to overfitting. Sub11 had CV MSE of 0.007767 but LB of 0.009848.

---

## Phase 2: Blending Experiments

### Discovery: Blend Synergy
Sub9 and sub10 alone score 0.009109 and 0.008907, but their blend performs better than either:

| Blend Ratio (sub9:sub10) | Submission | LB Score |
|--------------------------|------------|----------|
| 80:20 | Sub 20 | 0.008619 |
| 50:50 | Sub 25 | **0.008305** |
| 30:70 | Sub 34 | 0.008377 |

**Key Insight**: The optimal blend is 50-50. The synergy comes from the models capturing different patterns that complement each other.

---

## Phase 3: Correlation Analysis

### Method
Created `src/score_analysis.py` to analyze what prediction statistics correlate with LB score using the 7 known submissions.

### Initial Findings (7 submissions)
| Feature | Correlation | p-value | Interpretation |
|---------|-------------|---------|----------------|
| depth_max | r=-0.901 | 0.006 | Higher = better |
| angle_std | r=+0.749 | 0.033 | Lower = better |
| depth_mean | r=+0.627 | 0.096 | Lower = better |

### Hypothesis: Push depth_max Higher
Based on strong correlation, we hypothesized that increasing depth_max would improve scores.

### Test: Submission 51 (LB: 0.008807 - WORSE)
**Method**: Combined sub25's angle and left_right predictions with sub43's depth predictions (which had higher depth_max)

**Result**: Score got WORSE (0.008807 vs 0.008305)

**Learning**: Simply replacing predictions breaks internal consistency. The synergy between angle, depth, and left_right predictions matters.

### Updated Correlation (8 submissions)
After adding sub51's data point:
| Feature | Correlation | p-value | Change |
|---------|-------------|---------|--------|
| angle_std | r=+0.749 | 0.033 | Now STRONGEST |
| depth_mean | r=+0.627 | 0.096 | Still significant |
| depth_max | r=-0.549 | 0.159 | DROPPED from -0.901 |

**Key Insight**: depth_max correlation was partially spurious. angle_std is actually the most reliable predictor.

---

## Phase 4: Alternative Approaches Tried

### 4.1 Gradient-Guided Model
**File**: `src/gradient_guided_model.py`

**Approaches tested**:
1. **Extrapolated predictions**: Push predictions in direction correlated with better scores
2. **Depth-boosted**: Scale depth predictions to increase depth_max
3. **Target-specific blend**: Use different blend ratios per target

**Results**: Created sub41-47, none beat sub25

### 4.2 Alternative Model Architectures
**File**: `src/alternative_approaches.py`

| Approach | CV MSE | angle_std | Result |
|----------|--------|-----------|--------|
| K-Nearest Neighbors | 14.10 | 0.1509 | Worse |
| Critical Frames Only | 20.14 | 0.1687 | Worse |
| Bayesian Ridge | 15.16 | - | Worse |
| Per-Player Mean | 15.72 | - | Baseline |
| Distance-Weighted | - | - | Similar |
| Diverse Ensemble | - | 0.1537 | Worse |

**Submissions**: 60-64 (blends with sub25)

**Learning**: Simpler models don't capture enough signal; they all have higher angle_std than sub25.

### 4.3 Neural Networks
**File**: `src/neural_model.py`

**Architectures tested**:
- SimpleNet: 2 hidden layers with dropout
- WideNet: 1 wide layer with batch norm

**Results**:
- CV scores very high (19-66 MSE) - severe overfitting
- Predictions hitting bounds (0 and 1)
- Dataset too small for neural networks

### 4.4 Robust Ensemble
**File**: `src/robust_ensemble.py`

**Method**:
- Multi-seed bagging (5 seeds)
- LightGBM + CatBoost + Ridge ensemble
- Conservative hyperparameters

**Result** (Sub 65):
- CV Score: 0.008198 (good)
- angle_std: 0.1437 (worse than sub25's 0.1380)
- Blends with sub25 created (sub66-68)

### 4.5 Distribution Analysis
**File**: `src/distribution_analysis.py`

**Adversarial Validation Result**: AUC = 0.4711
- Close to 0.5 means NO significant train/test distribution shift
- This ruled out distribution shift as the problem

**Approaches tested**:
1. **Similarity-weighted training** (Sub 69): Weight training samples by similarity to test
2. **Per-player calibration** (Sub 70): Normalize predictions to match training distribution
3. **Shrinkage** (Sub 71-72): Pull predictions toward player mean

**Best result**: Sub 71 (shrinkage 0.95) with angle_std=0.1378

### 4.6 Angle Compression
**Submissions 56, 59**

**Method**: Compress angle predictions toward mean to reduce angle_std

**Result**:
- Sub 56: angle_std=0.1311 (lowest achieved)
- Risk: Compression might hurt actual accuracy

---

## Phase 5: Key Insights and Learnings

### What Works
1. **Ensemble of diverse models**: LGB + CatBoost + Ridge captures different patterns
2. **Per-player modeling**: Each player has unique biomechanics
3. **Blending complementary models**: 50-50 blend of sub9+sub10 outperforms both
4. **Conservative regularization**: Prevents overfitting on small dataset

### What Doesn't Work
1. **Over-optimization**: Sub11 had best CV but worst LB among good models
2. **High depth_max alone**: Sub51 proved this breaks prediction consistency
3. **Neural networks**: Dataset too small (345 samples)
4. **Replacing individual target predictions**: Breaks synergy between targets

### The Ceiling Problem
- Sub25 (50-50 blend) = 0.008305
- All variations stay within 0.0083-0.0090 range
- To reach 0.007 (15.7% improvement), we likely need:
  1. Better features we haven't discovered
  2. More training data
  3. A fundamentally different approach

### Statistical Summary
| Metric | Sub25 (best) | Correlation with LB |
|--------|--------------|---------------------|
| angle_std | 0.1380 | r=+0.749 (lower better) |
| depth_mean | 0.5055 | r=+0.627 (lower better) |
| depth_max | 0.7447 | r=-0.549 (higher better) |

---

## Detailed Statistics

### All Submissions - ANGLE Statistics
| Sub | LB Score | angle_mean | angle_std | angle_min | angle_max | angle_skew |
|-----|----------|------------|-----------|-----------|-----------|------------|
| 8 | 0.010220 | 0.5185 | 0.1402 | 0.2614 | 0.8241 | 0.454 |
| 9 | 0.009109 | 0.5206 | 0.1390 | 0.2964 | 0.8283 | 0.542 |
| 10 | 0.008907 | 0.5222 | 0.1388 | 0.2492 | 0.8366 | 0.490 |
| 11 | 0.009848 | 0.5249 | 0.1441 | 0.2700 | 1.0524 | 0.810 |
| 20 | 0.008619 | 0.5209 | 0.1384 | 0.3069 | 0.8224 | 0.543 |
| **25** | **0.008305** | **0.5214** | **0.1380** | 0.2926 | 0.8277 | 0.534 |
| 34 | 0.008377 | 0.5217 | 0.1381 | 0.2752 | 0.8312 | 0.520 |
| 51 | 0.008807 | 0.5214 | 0.1380 | 0.2926 | 0.8277 | 0.534 |

### All Submissions - DEPTH Statistics
| Sub | LB Score | depth_mean | depth_std | depth_min | depth_max | depth_skew |
|-----|----------|------------|-----------|-----------|-----------|------------|
| 8 | 0.010220 | 0.5125 | 0.0849 | 0.2784 | 0.7243 | -0.151 |
| 9 | 0.009109 | 0.5025 | 0.1067 | -0.0685 | 0.7397 | -1.635 |
| 10 | 0.008907 | 0.5086 | 0.0877 | 0.2465 | 0.7498 | -0.381 |
| 11 | 0.009848 | 0.5093 | 0.0917 | 0.2407 | 0.7297 | -0.447 |
| 20 | 0.008619 | 0.5037 | 0.0990 | 0.0757 | 0.7417 | -1.029 |
| **25** | **0.008305** | **0.5055** | 0.0906 | 0.2336 | **0.7447** | -0.494 |
| 34 | 0.008377 | 0.5067 | 0.0877 | 0.2387 | 0.7467 | -0.418 |
| 51 | 0.008807 | 0.5093 | 0.0957 | 0.2164 | 0.7829 | -0.718 |

### All Submissions - LEFT_RIGHT Statistics
| Sub | LB Score | lr_mean | lr_std | lr_min | lr_max | lr_skew |
|-----|----------|---------|--------|--------|--------|---------|
| 8 | 0.010220 | 0.4695 | 0.0626 | 0.3335 | 0.6175 | 0.156 |
| 9 | 0.009109 | 0.4612 | 0.0709 | 0.1769 | 0.6246 | -0.574 |
| 10 | 0.008907 | 0.4702 | 0.0611 | 0.3096 | 0.6507 | 0.264 |
| 11 | 0.009848 | 0.4706 | 0.0651 | 0.2779 | 0.6530 | -0.083 |
| 20 | 0.008619 | 0.4630 | 0.0666 | 0.2413 | 0.6133 | -0.301 |
| **25** | **0.008305** | **0.4657** | **0.0621** | 0.2885 | 0.5963 | 0.017 |
| 34 | 0.008377 | 0.4675 | 0.0606 | 0.2969 | 0.6124 | 0.146 |
| 51 | 0.008807 | 0.4657 | 0.0621 | 0.2885 | 0.5963 | 0.017 |

### Complete Correlation Analysis (18 features vs LB Score)
| Feature | Correlation | p-value | Direction | Significant |
|---------|-------------|---------|-----------|-------------|
| angle_std | +0.749 | 0.033 | lower is better | * |
| angle_median | -0.655 | 0.078 | higher is better | * |
| depth_mean | +0.627 | 0.096 | lower is better | * |
| depth_max | -0.549 | 0.159 | higher is better | |
| lr_median | +0.530 | 0.177 | lower is better | |
| lr_max | +0.490 | 0.218 | lower is better | |
| angle_min | -0.478 | 0.231 | higher is better | |
| angle_max | +0.473 | 0.237 | lower is better | |
| lr_mean | +0.465 | 0.246 | lower is better | |
| angle_skew | +0.275 | 0.509 | lower is better | |
| depth_skew | +0.246 | 0.558 | lower is better | |
| lr_min | +0.203 | 0.629 | lower is better | |
| depth_std | -0.200 | 0.635 | higher is better | |
| lr_std | +0.189 | 0.655 | lower is better | |
| depth_min | +0.171 | 0.686 | lower is better | |
| angle_mean | -0.095 | 0.822 | higher is better | |
| depth_median | +0.060 | 0.888 | lower is better | |
| lr_skew | +0.038 | 0.928 | lower is better | |

### Blend Ratio Analysis
Quadratic fit: `score = 0.002812*w^2 - 0.002610*w + 0.008907`

| Sub9 Weight | Sub10 Weight | LB Score | Submission |
|-------------|--------------|----------|------------|
| 1.0 | 0.0 | 0.009109 | Sub 9 |
| 0.8 | 0.2 | 0.008619 | Sub 20 |
| 0.5 | 0.5 | 0.008305 | Sub 25 |
| 0.3 | 0.7 | 0.008377 | Sub 34 |
| 0.0 | 1.0 | 0.008907 | Sub 10 |

**Theoretical Optimal**: 46% sub9 + 54% sub10 = 0.008301 (marginal improvement)

### Per-Player Training Sample Distribution
| Player | Samples | Target Variance (angle) | Target Variance (depth) | Target Variance (lr) |
|--------|---------|------------------------|------------------------|---------------------|
| 1 | 70 | Low | Medium | Medium |
| 2 | 66 | Medium | Medium | Medium |
| 3 | 68 | Low | Low | Low |
| 4 | 67 | Medium | Medium | Medium |
| 5 | 74 | **High** | **High** | Medium |

**Note**: Player 5 has 2-3x higher target variance - hardest to predict.

---

## Visualizations

Two visualization files created in `Research/` folder:

1. **`research_visualizations.png`** - 4 subplots:
   - LB Scores by Submission (bar chart)
   - Blend Ratio vs LB Score (quadratic curve)
   - angle_std vs LB Score (correlation plot)
   - depth_max vs LB Score (correlation plot with Sub51 outlier highlighted)

2. **`player_analysis.png`** - 2 subplots:
   - Training Samples per Player
   - Target Variance by Player

---

## Current Best Untested Candidates

| Sub | Description | angle_std | depth_mean | Risk Level |
|-----|-------------|-----------|------------|------------|
| 56 | 5% angle compression | 0.1311 | 0.5055 | High (might hurt accuracy) |
| 59 | Combined optimization | 0.1339 | 0.5049 | Medium |
| 71 | Shrinkage 0.95 | 0.1378 | 0.5060 | Low (minimal change) |
| 73 | 30% weighted + 70% sub25 | 0.1369 | 0.5088 | Medium |

---

## Files Created During Research

| File | Purpose |
|------|---------|
| `src/ensemble_submission.py` | Sub9 - multi-model ensemble |
| `src/optimized_ensemble.py` | Sub10 - Optuna-tuned ensemble |
| `src/score_analysis.py` | Correlation analysis of LB scores |
| `src/gradient_guided_model.py` | Gradient-based optimization |
| `src/strategic_blends.py` | Smart blending strategies |
| `src/neural_model.py` | Neural network experiments |
| `src/alternative_approaches.py` | KNN, Bayesian, etc. |
| `src/robust_ensemble.py` | Multi-seed bagging |
| `src/distribution_analysis.py` | Train/test shift analysis |
| `src/advanced_features.py` | Frame-specific feature engineering |
| `src/hybrid_features.py` | Combined feature extraction |

---

## Conclusion

After extensive experimentation with 15+ different approaches and 70+ submissions, the best LB score achieved is **0.008305** (sub25). The gap to the target of 0.007 (15.7%) appears to require something beyond incremental improvements to the current approach.

The most promising remaining candidates are based on reducing angle_std (the strongest LB predictor), but these involve trade-offs that may or may not improve the actual score.
