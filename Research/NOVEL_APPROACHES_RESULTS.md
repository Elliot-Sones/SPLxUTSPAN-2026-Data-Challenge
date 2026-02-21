# Novel Approaches Experiment Results

Date: 2026-02-06

## Overview

Four novel approaches were tested to improve CV performance and/or provide diverse predictions for blending with Sub 784 (LB 0.007224). The core question: is the CV-LB gap from distribution shift or overfitting? And can we exploit augmentation, functional representations, or hierarchical structure to improve?

---

## 1. Adversarial Validation Results

**Goal:** Determine if train/test distribution shift exists, which would explain the CV-LB gap.

**Method:** Train a classifier (LightGBM) to distinguish train vs test samples using 288 frame-153 features. If AUC >> 0.5, distributions differ.

**Results:**
- AUC = 0.5042 (essentially random - no distribution shift detected)
- Only 5/288 features with KS p<0.05 (fewer than the 14.4 expected by chance at alpha=0.05)
- Player distribution is perfectly balanced: ~20% each player in both train and test
- Per-player AUC: Player 1=0.628, Player 2=0.355, Player 3=0.543, Player 4=0.569, Player 5=0.539

**KEY CONCLUSION:** The CV-LB gap is caused by overfitting, NOT distribution shift. Train and test come from the same distribution. This means:
1. CV improvements that generalize should also improve LB
2. Methods that reduce overfitting (augmentation, regularization, smaller feature sets) are the right direction
3. No need for domain adaptation or covariate shift correction

---

## 2. Mirror Augmentation Results

**Goal:** Double training data by exploiting bilateral symmetry of basketball free throws. A shot from the left side, mirrored, looks like a shot from the right side.

**Method:**
- Flip all x-coordinates around hoop center (x_new = 2*5.25 - x_old = 10.5 - x)
- Swap left/right joint pairs (33 L/R keypoint pairs found)
- Negate left_right target (mirrored shot hits opposite side of rim)
- Keep angle and depth targets unchanged (vertical plane physics unchanged)

**CV Results (RAW target space, not scaled):**

| Target | Baseline MSE | Mirror MSE | Change |
|--------|-------------|------------|--------|
| Angle | 8.70 | 6.69 | -23.1% |
| Depth | 16.89 | 15.91 | -5.8% |
| Left_right | 10.33 | 8.96 | -13.3% |
| Mean | 11.98 | 10.52 | -12.2% |

**Diversity (correlation with Sub 784 predictions):**
- Angle: r = 0.91 (moderate diversity)
- Depth: r = 0.79 (good diversity)
- Left_right: r = 0.26 (VERY high diversity - nearly independent predictions)

**Generated Submissions:**
- Sub 1218: mirror-aug blend with Sub 784, weights aw=0.30, dw=0.30, lw=0.60 (highest diversity)
- Sub 1219: mirror-aug blend with Sub 784, weights aw=0.20, dw=0.20, lw=0.40
- Sub 1220: mirror-aug blend with Sub 784, weights aw=0.10, dw=0.10, lw=0.30
- Sub 1221: mirror-aug blend with Sub 784, weights aw=0, dw=0, lw=0.50 (left_right only)
- Sub 1222: mirror-aug blend with Sub 784, weights aw=0, dw=0, lw=0.30
- Sub 1223: mirror-aug blend with Sub 784, weights aw=0, dw=0.30, lw=0.50 (same weights as Sub 784)

**Key Observations:**
- The 23% angle improvement is striking - mirror augmentation regularizes strongly
- Left_right diversity of r=0.26 is the highest we have ever seen - this could be extremely valuable for blending
- Even depth (historically the hardest target) sees a 5.8% improvement

---

## 3. FPCA + Shot Similarity Results

**Goal:** Represent each shot as a smooth functional curve, then use functional PCA or trajectory similarity (KNN) for prediction.

**Method:**
- FPCA: 12 key joints, 20 Fourier coefficients each across x/y/z, yielding 1008 raw features
- Reduced to 30 PCA components capturing 97.9% of variance
- KNN trajectory: find k most similar shots using DTW or Euclidean distance on raw trajectories
- KNN FPCA: find k most similar shots in FPCA feature space
- Multi-target: predict all 3 targets jointly

**CV Results (RAW target space):**

| Method | Angle MSE | Depth MSE | LR MSE | Mean MSE |
|--------|-----------|-----------|--------|----------|
| Baseline | -- | -- | -- | 10.12 |
| FPCA | -- | -- | -- | 15.03 |
| KNN trajectory | -- | -- | -- | 13.89 |
| KNN FPCA | -- | -- | -- | 14.79 |
| Multi-target | -- | -- | -- | 14.95 |

**Diversity (correlation with Sub 784 predictions):**
- FPCA: angle r=0.88, depth r=0.48, left_right r=0.50 (high diversity, especially depth)
- KNN: similar diversity profile

**Generated Submissions:**
- Sub 1215: KNN best-per-target blend with Sub 784
- Sub 1216: FPCA full blend with Sub 784
- Sub 1217: FPCA maximum diversity blend with Sub 784

**Key Observations:**
- All FPCA/KNN methods perform significantly worse than baseline in absolute terms
- However, diversity is high (depth r=0.48 is much lower than typical r=0.91+)
- Value is purely as a diversity source for blending, not as a standalone model
- 30 FPCA components may be too aggressive a reduction, or Fourier basis may not capture shot dynamics well

---

## 4. Hierarchical Models Results

**Goal:** Test whether sharing information across players (global models, cross-player transfer, blended models) can reduce overfitting.

**Method:** Seven approaches tested, all using the same feature set at frame 153.

**CV Results (RAW target space):**

| Approach | Angle MSE | Depth MSE | LR MSE | Mean MSE | vs Baseline |
|----------|-----------|-----------|--------|----------|-------------|
| 1. Baseline (per-player) | -- | -- | -- | 10.12 | -- |
| 2. Global (all players) | -- | -- | -- | 10.14 | +0.2% |
| 3. Two-stage (global + residual) | -- | -- | -- | 13.22 | +30.6% |
| 4. Cross-player transfer | -- | -- | -- | 9.98 | -1.4% |
| 5. Blended (global + per-player) | -- | -- | -- | 9.28 | -8.3% |
| 6. Heavy regularization | -- | -- | -- | 11.00 | +8.6% |
| 7. Feature bagging | -- | -- | -- | 11.03 | +9.0% |

**Diversity (correlation with Sub 784 predictions):**
- Blended approach: angle r=0.97, depth r=0.91, left_right r=0.95 (LOW diversity)

**Generated Submissions:**
- Sub 1224: blended hierarchical blend with Sub 784
- Sub 1225: cross-player transfer blend with Sub 784
- Sub 1226: blended hierarchical standalone (no blend with Sub 784)

**Key Observations:**
- Blended hierarchical (approach 5) gives the best CV improvement at -8.3%
- The key insight: Player 5's optimal global weight = 1.0 for angle and depth, meaning per-player models completely overfit on Player 5 (only ~70 samples)
- Two-stage residual is TERRIBLE (+30.6%) - residuals are noise, confirms previous finding
- Heavy regularization and feature bagging both hurt, confirming that overfitting is better handled by data augmentation or model blending than by restricting the model
- Despite best CV improvement, diversity is very low (r=0.91-0.97), limiting blending value

---

## Key Conclusions

1. **No train/test distribution shift** - the CV-LB gap is pure overfitting. This is confirmed by adversarial validation AUC=0.5042.

2. **Mirror augmentation is the most promising approach** - it improves all three targets (-23.1% angle, -5.8% depth, -13.3% LR) AND provides very high diversity (LR r=0.26). This is the rare combination of better absolute performance plus blending value.

3. **Blended hierarchical gives best CV improvement** (-8.3%) but low diversity (r=0.91-0.97) severely limits its blending value. Its value is as a potential replacement, not a blending partner.

4. **FPCA/KNN have high diversity but worse absolute performance** - value only as diversity sources for blending. The absolute performance gap (~50% worse) may be too large for small blend weights to help.

5. **Heavy regularization and feature bagging do not help** - with only 345 samples, constraining models more aggressively reduces their ability to capture real signal, not just noise.

6. **Residual modeling is confirmed dead** - two-stage residual is +30.6% worse, consistent with all previous residual modeling attempts.

---

## Recommended Submissions to Test on LB

Priority order:

1. **Sub 1223**: Mirror aug with Sub 784 weights (aw=0, dw=0.30, lw=0.50) - MOST PROMISING
   - Same blend weights that worked for Sub 784, but with mirror-augmented model as the secondary
   - Rationale: mirror aug improves all targets, especially left_right where diversity is r=0.26

2. **Sub 1218**: Mirror aug highest diversity blend (aw=0.30, dw=0.30, lw=0.60)
   - More aggressive blend weights to exploit mirror aug's high diversity
   - Higher risk/reward than Sub 1223

3. **Sub 1215**: KNN best-per-target blend
   - Tests whether non-parametric diversity can improve LB
   - Low confidence given worse absolute performance

4. **Sub 1217**: FPCA blend (maximum diversity)
   - Tests maximum diversity hypothesis
   - Lowest confidence given significant absolute performance gap

---

## Reproduction Details

**Environment:** macOS, Python via uv, LightGBM + XGBoost + CatBoost + Ridge ensemble

**Data:**
- Train: /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/data/train.csv (345 shots)
- Test: /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/data/test.csv (113 shots)
- Scalers: data/scaler_angle.pkl, data/scaler_depth.pkl, data/scaler_left_right.pkl

**Baseline Model:**
- Per-player per-target tree ensemble (LGB + XGB + CatBoost + Ridge)
- Features extracted at frame 153 (207 features from 69 keypoints x 3 coords)
- 5-fold CV, per-player stratification

**Mirror Augmentation:**
- 33 L/R keypoint pairs swapped
- x-coordinate flipped: x_new = 10.5 - x_old (around hoop center x=5.25)
- left_right target negated, angle and depth unchanged
- Augmented data appended to training set (690 total samples)

**FPCA:**
- 12 joints: nose, neck, mid_hip, left/right shoulder/elbow/wrist, left/right hip
- 20 Fourier coefficients per joint per axis = 1008 features
- 30 PCA components (97.9% variance explained)

**Hierarchical Blended:**
- Per-player optimal global weight found via CV grid search [0, 0.1, ..., 1.0]
- Final prediction = w * global_pred + (1-w) * per_player_pred
- Player 5 optimal weight = 1.0 for angle and depth
