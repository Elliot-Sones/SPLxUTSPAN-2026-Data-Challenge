# Plateau Breakthrough Research - SPL UTSpan 2026 Data Challenge

## Executive Summary

**Date**: 2026-02-09
**Current Best**: Sub 1640, LB 0.006698 (10% LASSO + 90% Sub 1350)
**Target**: LB < 0.005 (25.2% improvement needed)
**Gap**: 0.001698 MSE
**Constraint**: 345 train shots, 113 test shots, 5 players

Based on comprehensive Kaggle research for small tabular datasets and analysis of what has been attempted, this document proposes 8 novel approaches specifically designed to break through the 0.0066-0.0067 plateau.

---

## Problem Context

### What Has Failed (High Correlation r>0.85)
1. Feature selection (LASSO, per-target) - LB 0.006785-0.006814
2. OOF stacking (120 models) - LB 0.006716
3. Per-example V2 (bandwidth/alpha tuning) - LB 0.006789
4. Physics modeling - definitively impossible (3.6x velocity gap)
5. Biomechanical features - LB 0.007794 (good CV, bad LB)
6. Multi-task learning - LB 0.006803 (CV 0.005093 but 33.6% gap)
7. Ball trajectory features - LB 0.006918
8. Mirror augmentation - LB 0.011905 (catastrophic)
9. Temporal dynamics - LB 0.007528
10. Gaussian Process - r=0.907-0.966 with Sub 1350
11. External data (SPL Open Data) - no improvement

### Core Constraint
**CV-LB gap is 33-81%**: LOO/LOPO CV is systematically optimistic. Models that look excellent in CV often fail on LB. This is the primary challenge.

### What Works
- Per-example locally weighted Ridge regression (Gaussian kernel, bandwidth=0.5 quantile, alpha=10)
- LASSO stability selection at small blend weights (10%)
- Target-specific frames (angle=153, depth=150, LR=170)
- 198 hoop-relative coordinates + 15 PLS components = 213 features
- Blend weights: dw=0.30, lw=0.50 with Sub 784

---

## Kaggle Research Findings (2024-2025)

### 1. Stacking Best Practices

**Source**: [Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/), [OOF Meta-modelling](https://www.kaggle.com/code/amoghjrules/and-the-stacking-continues-oof-meta-modelling)

**Key Findings**:
- Multi-level stacking (3 levels) won Kaggle April 2025 Playground
- Use k-fold cross-validated scores instead of training data predictions to reduce overfitting
- OOF predictions can be used AS FEATURES in meta-model (not just blended)
- RFE (Recursive Feature Elimination) + SHAP identifies salient meta-features
- Stacking typically restricted to 1-2 meta-layers due to overfitting risk

**Critical for small data**:
- With 345 samples, even 2-level stacking risks overfitting
- Must use aggressive regularization (Ridge alpha=100+)
- Need DIVERSE base models (r<0.70) for stacking to work

### 2. Small Sample Deep Learning

**Source**: [Survey on Deep Tabular Learning](https://arxiv.org/abs/2410.12034), [Tabular Deep Learning](https://arxiv.org/html/2407.00956v2)

**Key Findings**:
- Meta-learned foundation models (Hollmann et al., 2023) outperform GBDTs in small datasets
- TANGOS encourages neuron specialization via gradient attributions
- ExcelFormer uses semi-permeable attention to constrain less informative features
- Graph-based models (GNN4TDL, GANDALF) mitigate overfitting through advanced regularization
- TabDDPM generates synthetic data via diffusion to address data scarcity

**Critical for small data**:
- Standard neural nets fail (we confirmed: neural MTL 4.4-14x worse)
- Need specialized architectures designed for <1000 samples
- Diffusion-based synthetic data generation is promising

### 3. Target Encoding for Small Datasets

**Source**: [Target Encoding](https://www.kaggle.com/code/ryanholbrook/target-encoding), [Regularized Target Encoding](https://link.springer.com/article/10.1007/s00180-022-01207-6)

**Key Findings**:
- Great for high-cardinality features (Player ID has 5 categories)
- Smoothing prevents overfitting: blend in-category average with overall average
- Rare categories get less weight (Player 5 has only 74 samples)
- Use 4-5 fold cross-validation to prevent leakage
- Frequency encoding also performed well in benchmarks

**Critical for small data**:
- Smoothed target encoding: `encoded = (count * category_mean + m * global_mean) / (count + m)`
- m (smoothing parameter) typically 10-50 for small datasets
- Combined with GroupKFold to prevent player leakage

### 4. Power Transforms and Quantile Normalization

**Source**: [Power Transforms](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html), [Ordered Quantile Normalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9042069/)

**Key Findings**:
- Box-Cox (positive values only) and Yeo-Johnson (positive/negative) make data more Gaussian
- Optimal lambda parameter estimated per feature using maximum likelihood
- Quantile normalization (ORQ) is semiparametric approach using ranks + interpolation
- bestNormalize framework compares Box-Cox, Yeo-Johnson, Lambert WxF, ORQ, log, sqrt transformations

**Critical for small data**:
- Power transforms stabilize variance and reduce skewness
- Can improve Ridge regression by making features more Gaussian
- Apply per-feature, not globally

### 5. Nested Cross-Validation

**Source**: [Nested CV scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [Nested CV for Small Data](https://pubmed.ncbi.nlm.nih.gov/40039298/)

**Key Findings**:
- Double loop: outer loop for model evaluation, inner loop for hyperparameter tuning
- Eliminates test-set information leakage into model selection
- Prevents overoptimistic error estimates (addresses our 33-81% CV-LB gap)
- 2024 study: higher predictive power on small data without overfitting

**Critical for small data**:
- Outer loop: 5-fold GroupKFold (leave-one-player-out equivalent)
- Inner loop: 3-fold for hyperparameter search
- Computationally expensive but prevents leakage

### 6. Bayesian Optimization for Ensemble Weights

**Source**: [Bayesian Optimization](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_4.xhtml), [Ensemble Weight Optimization](https://www.sciencedirect.com/science/article/pii/S0957417425019001)

**Key Findings**:
- Nelder-Mead optimization found optimal ensemble weights [0.32, 0.142, 0.434, 0.229]
- Bayesian methods (GP + Expected Improvement) more efficient than grid search
- Weighted voting where each model's vote weighted by pre-defined accuracy
- Neural networks can also learn optimal ensemble weights

**Critical for small data**:
- Optimize over OOF predictions (not validation set) to use all data
- Use nested CV: outer loop for evaluation, inner loop for weight optimization
- Search space: [0,1]^N where N is number of models to blend

### 7. Kaggle Shake-up Prevention

**Source**: [Kaggle Shake-up Fundamentals](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3bc8), [CV Strategy](https://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p2.html)

**Key Findings**:
- Public LB typically 20-40% of test set, private LB 60-80%
- Winners often ranked lower on public LB (overfitting to public test set)
- Cross-validation must match competition structure (GroupKFold for player-grouped data)
- Trust CV over public LB when they conflict

**Critical for small data**:
- Our CV-LB gap (33-81%) indicates CV is too optimistic OR models overfit to public test
- Should NOT optimize for public LB - it's unreliable
- Focus on robust CV + diverse ensemble for private LB

### 8. Feature Accumulation in Deep Stacking

**Source**: [RocketStack Framework](https://arxiv.org/html/2506.16965), [Stacking Guide](https://datasciblog.github.io/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

**Key Findings**:
- Dimensionality increases when predictions propagated through stacking layers
- Concatenating OOF vectors from all learners forms meta-feature matrix
- RFE filters redundant meta-features to reduce training cost
- Barriers: feature accumulation, training inefficiency, model overfitting

**Critical for small data**:
- With 345 samples, adding 100+ meta-features (OOF predictions) risks severe overfitting
- Must use aggressive feature selection (LASSO, RFE, SHAP)
- Prefer shallow stacking (1 layer) with strong regularization

---

## Proposed Breakthrough Approaches

### Approach 1: OOF Predictions AS FEATURES (Not Blending)

**Theory**: Instead of post-hoc blending, use OOF predictions as additional features in per-example regression. This allows the locally weighted model to learn when each base model is reliable for specific shot types.

**Implementation**:
1. Generate 20+ diverse base models (different algorithms, feature spaces, seeds)
2. Compute OOF predictions for each base model using GroupKFold
3. Concatenate: [198 HC features, 15 PLS components, 20 OOF predictions] = 233 features
4. Apply per-example locally weighted Ridge regression on 233 features
5. Use nested CV: outer GroupKFold for evaluation, inner for bandwidth/alpha tuning

**Why it might work**:
- OOF predictions capture learned patterns from different models
- Locally weighted regression can learn which models to trust for each shot type
- 233 features still manageable for Ridge with strong regularization (alpha=100)
- Different from our failed "OOF stacking" (Sub 1430) which just blended post-hoc

**Risk**: 20 OOF features may still overfit. Mitigation: use LASSO to select top 5 OOF features.

**Estimated effort**: 4 hours (generate base models + implement + validate)
**Probability of improvement**: 35%

---

### Approach 2: Smoothed Target Encoding for Player ID

**Theory**: Player ID is high-cardinality categorical (5 players) with imbalanced samples (Player 5: 74, others: 67-69). Smoothed target encoding can capture player-specific biases without overfitting.

**Implementation**:
1. For each target (angle, depth, left_right), compute smoothed encoding:
   - `encoded_angle[player] = (count * player_mean_angle + 10 * global_mean_angle) / (count + 10)`
   - Smoothing parameter m=10 (recommended for small data)
2. Add 3 features: [encoded_angle_player, encoded_depth_player, encoded_lr_player]
3. Use 5-fold GroupKFold during encoding to prevent leakage
4. Concatenate with existing 213 features → 216 features
5. Apply per-example locally weighted Ridge regression

**Why it might work**:
- Captures player-specific biases (Player 2 shoots deeper, Player 5 more left)
- Smoothing prevents overfitting on Player 5 (only 74 samples)
- Different from "player dummy variables" which are linear and don't capture target relationships
- Regularized encoding is less prone to leakage than raw player means

**Risk**: May capture CV patterns that don't generalize to test. Mitigation: very conservative blend (5%).

**Estimated effort**: 2 hours
**Probability of improvement**: 25%

---

### Approach 3: Ordered Quantile Normalization (ORQ) + Power Transforms

**Theory**: Our features may have non-Gaussian distributions with skewness. Power transforms (Yeo-Johnson) can make features more Gaussian, improving Ridge regression performance.

**Implementation**:
1. Apply Yeo-Johnson transform to all 213 features per-sample:
   - Estimate optimal lambda per feature using MLE
   - Transform training data
   - Apply same transform to test data
2. Apply ORQ (Ordered Quantile Normalization):
   - Rank-based transformation to enforce Gaussian distribution
   - Uses original values + ranks + interpolation
3. Concatenate: [213 original features, 213 transformed features] = 426 features
4. Apply LASSO (alpha=0.01) to select top 150 features
5. Apply per-example locally weighted Ridge regression on 150 features

**Why it might work**:
- Ridge regression assumes Gaussian features - power transforms satisfy this
- ORQ shown to work well in cross-validation era (2024 research)
- Transformed features may capture non-linear patterns missed by raw features
- Different feature space → potential for low correlation with Sub 1350

**Risk**: Doubling features to 426 may overfit. Mitigation: aggressive LASSO selection.

**Estimated effort**: 3 hours
**Probability of improvement**: 30%

---

### Approach 4: Nested Cross-Validation with Bayesian Optimization

**Theory**: Our CV-LB gap (33-81%) is caused by hyperparameter tuning on CV scores (information leakage). Nested CV eliminates this by treating hyperparameter optimization as part of the model.

**Implementation**:
1. Outer loop: 5-fold GroupKFold (each fold leaves out 1 player)
2. Inner loop: 3-fold for Bayesian optimization of (bandwidth, alpha, lw, dw)
   - Search space: bandwidth=[0.3, 0.8], alpha=[5, 50], lw=[0.3, 0.7], dw=[0.1, 0.5]
   - Use GP + Expected Improvement (Bayesian optimization)
   - 30 iterations per inner fold
3. For each outer fold:
   - Optimize hyperparameters on 4 training folds (inner loop)
   - Evaluate on held-out fold (outer loop)
4. Report honest nested CV score (unbiased estimate of LB)
5. Retrain on full data with optimal hyperparameters

**Why it might work**:
- Eliminates information leakage from hyperparameter tuning
- Provides honest estimate of LB (may reveal Sub 1350 is overfit)
- Bayesian optimization more efficient than grid search (345 samples precious)
- Different from our current LOO CV (which still leaks info via hyperparameter selection)

**Risk**: Computationally expensive (5 outer x 3 inner x 30 trials = 450 model fits). May reveal Sub 1350 is already optimal.

**Estimated effort**: 5 hours (implement Bayesian optimization + nested CV)
**Probability of improvement**: 20% (may just confirm Sub 1350 is optimal)

---

### Approach 5: TabDDPM Synthetic Data Generation

**Theory**: 345 samples insufficient for complex models. Diffusion-based synthetic data generation (TabDDPM) can create realistic synthetic shots, increasing training data to 1000+ samples.

**Implementation**:
1. Train TabDDPM (Tabular Denoising Diffusion Probabilistic Model):
   - Input: 213 features (198 HC + 15 PLS)
   - Output: 3 targets (angle, depth, left_right)
   - 1000 diffusion steps, Gaussian noise schedule
   - Conditional generation: given Player ID, generate shot features
2. Generate 655 synthetic shots (1000 total with 345 real)
3. Train per-example locally weighted Ridge on 1000 samples
4. Test on real 113 test shots

**Why it might work**:
- TabDDPM specifically designed for small tabular datasets (2024 research)
- Generative model learns underlying distribution, fills in gaps
- More data allows weaker regularization (lower alpha)
- Conditional generation respects player-specific patterns

**Risk**: Synthetic shots may not be realistic (physics constraints violated). Generated data could introduce bias.

**Mitigation**:
- Validate synthetic shots have plausible physics (velocity 6-8 m/s)
- Train with varying synthetic ratios: 10%, 30%, 50%, 70%
- Blend synthetic-trained model conservatively (20%) with Sub 1350

**Estimated effort**: 8 hours (implement TabDDPM + validation)
**Probability of improvement**: 25%

---

### Approach 6: Meta-Learned Foundation Model (Hollmann et al. 2023 approach)

**Theory**: Standard neural nets fail on 345 samples. Meta-learning trains on multiple related tasks (other sports/motion capture datasets) then fine-tunes on basketball shots.

**Implementation**:
1. Collect meta-learning datasets:
   - SPL Open Data (125 shots, all misses) - use for kinematics learning
   - Human3.6M (motion capture) - use for pose dynamics learning
   - AMASS (motion capture) - use for skeleton structure learning
2. Pre-train TabNet on meta-tasks:
   - Task 1: Predict 3D keypoint velocities from poses (Human3.6M)
   - Task 2: Predict action class from pose sequence (AMASS)
   - Task 3: Predict shot outcome (make/miss) from kinematics (SPL)
3. Fine-tune TabNet on SPL UTSpan:
   - Freeze first 2 layers (general kinematics)
   - Fine-tune last 2 layers on 345 shots
   - Strong regularization (dropout=0.5, weight_decay=0.01)
4. Ensemble with Sub 1350 (20% meta-learned, 80% Sub 1350)

**Why it might work**:
- Hollmann et al. (2023) showed meta-learning outperforms GBDTs on <1000 samples
- Pre-training on motion capture teaches general kinematics
- Fine-tuning adapts to basketball-specific patterns
- Different learning paradigm than Ridge regression

**Risk**: Meta-learning datasets may have domain mismatch. Pre-training could introduce bad biases.

**Mitigation**:
- Validate meta-features transfer: compute correlation with SPL targets
- Use very conservative blend weight (10-20%)

**Estimated effort**: 12 hours (download datasets + implement meta-learning + validate)
**Probability of improvement**: 20%

---

### Approach 7: SHAP-Based Meta-Feature Selection for Deep Stacking

**Theory**: Our OOF stacking failed (Sub 1430 = 0.006782) because we blended post-hoc. Deep stacking with RFE + SHAP can identify which OOF predictions are most valuable.

**Implementation**:
1. Generate 50 diverse base models:
   - Different algorithms: LGB, XGB, CatBoost, Ridge, ElasticNet
   - Different feature spaces: HC-only, PLS-only, physics features, biomech features
   - Different hyperparameters: 5 seeds each
2. Compute OOF predictions for each model (GroupKFold)
3. Build meta-feature matrix: [213 original features, 50 OOF predictions] = 263 features
4. Apply SHAP feature importance:
   - Train Ridge on 263 features
   - Compute SHAP values for each feature
   - Select top 30 features by |SHAP value|
5. Apply RFE (Recursive Feature Elimination):
   - Start with 30 SHAP-selected features
   - Remove 1 feature at a time, retrain, evaluate CV
   - Keep top 15 features
6. Train per-example locally weighted Ridge on 15 meta-features

**Why it might work**:
- SHAP identifies which OOF predictions are truly informative (not redundant)
- RFE ensures selected features generalize (nested CV during elimination)
- 15 features << 345 samples, prevents overfitting
- Different from blind OOF stacking (uses interpretability to guide selection)

**Risk**: SHAP selection may overfit to CV. RFE is computationally expensive.

**Mitigation**:
- Use nested CV during RFE: outer GroupKFold for evaluation, inner for selection
- Validate selected features are diverse (low inter-correlation r<0.7)

**Estimated effort**: 6 hours (generate 50 models + SHAP + RFE)
**Probability of improvement**: 30%

---

### Approach 8: Uncertainty-Weighted Per-Example Regression

**Theory**: Per-example regression treats all training neighbors equally (Gaussian weighting by distance). MC Dropout showed angle uncertainty is well-calibrated (r=0.83). Weight neighbors by inverse uncertainty.

**Implementation**:
1. Train ensemble of 10 Ridge models with different seeds
2. For each training shot, compute prediction variance across 10 models:
   - `uncertainty[shot] = std([model1(shot), model2(shot), ..., model10(shot)])`
3. Per-example regression with uncertainty weighting:
   - For test shot i, find K=50 nearest training neighbors
   - Weight neighbor j by: `w[j] = exp(-dist[i,j]^2 / bandwidth^2) / uncertainty[j]`
   - Train Ridge on weighted neighbors
   - Predict test shot i
4. Rationale: High-uncertainty training shots are less reliable → downweight them

**Why it might work**:
- MC Dropout showed uncertainty is well-calibrated for angle (r=0.83)
- High-uncertainty shots are likely noisy/outliers → excluding them improves fit
- Different from standard Gaussian weighting (which ignores prediction confidence)
- Combines per-example + ensemble uncertainty quantification

**Risk**: Uncertainty may not be well-calibrated for depth/left_right (r=0.08). Downweighting uncertain shots may remove informative variance.

**Mitigation**:
- Apply uncertainty weighting only for angle (well-calibrated)
- Use standard Gaussian weighting for depth/left_right
- Blend conservatively (30%) with Sub 1350

**Estimated effort**: 4 hours (implement ensemble + uncertainty weighting)
**Probability of improvement**: 25%

---

## Prioritization and Testing Strategy

### Tier 1: High Priority (Test First)

1. **Approach 1: OOF Predictions AS FEATURES** (35% prob, 4 hours)
   - Novel: uses OOF as features, not post-hoc blending
   - Builds on proven per-example regression
   - Manageable risk (LASSO feature selection)

2. **Approach 7: SHAP Meta-Feature Selection** (30% prob, 6 hours)
   - Novel: interpretability-guided stacking
   - Addresses why OOF stacking failed (redundant features)
   - Proven in 2024 Kaggle competitions

3. **Approach 3: Power Transforms + ORQ** (30% prob, 3 hours)
   - Novel: different feature space (transformed)
   - Low risk (can blend conservatively)
   - Fast to implement

### Tier 2: Medium Priority (Test if Tier 1 Succeeds)

4. **Approach 8: Uncertainty-Weighted Regression** (25% prob, 4 hours)
   - Novel: combines per-example + uncertainty
   - Builds on proven MC Dropout calibration (r=0.83)
   - Moderate risk

5. **Approach 5: TabDDPM Synthetic Data** (25% prob, 8 hours)
   - Novel: generative data augmentation
   - High effort but proven in 2024 research
   - Validate synthetic shots are realistic

6. **Approach 2: Smoothed Target Encoding** (25% prob, 2 hours)
   - Novel: regularized player encoding
   - Low effort, low risk
   - May provide small incremental gain

### Tier 3: High Risk / High Effort (Test if Desperate)

7. **Approach 4: Nested CV + Bayesian Optimization** (20% prob, 5 hours)
   - May just confirm Sub 1350 is optimal
   - Expensive but provides honest estimate
   - Do this if we need to validate approach

8. **Approach 6: Meta-Learned Foundation Model** (20% prob, 12 hours)
   - Very high effort (12 hours)
   - Domain mismatch risk
   - Only if neural nets show promise

---

## Testing Protocol

### Phase 1: Rapid Validation (1 submission per approach)
1. Implement Approach 1 (OOF as features)
2. Generate Sub 1685, test on LB
3. If LB < 0.00667 → proceed to variants
4. If LB >= 0.00667 → move to Approach 7

### Phase 2: Optimization (if improvement found)
1. Hyperparameter sweep (Bayesian optimization)
2. Blend weight optimization (0.1, 0.2, 0.3, 0.4, 0.5)
3. Generate 5 submissions, test best 2

### Phase 3: Ensemble (if multiple approaches work)
1. Combine Approach 1 + Approach 7 (if both improve)
2. Optimize ensemble weights with nested CV
3. Generate final submission

### Expected Timeline
- Tier 1 (3 approaches): 13 hours total, 3 submissions
- Tier 2 (3 approaches): 18 hours total, 3 submissions
- Tier 3 (2 approaches): 17 hours total, 2 submissions
- **Total**: 48 hours, 8 submissions

---

## Risk Mitigation

### If All Approaches Fail (LB >= 0.00667)
1. Accept Sub 1640 (0.006698) as near-optimal
2. Focus on private LB strategy:
   - Diversify submissions (select models with r<0.85)
   - Trust CV over public LB
   - Don't overfit to public test set

### If One Approach Succeeds (LB = 0.00660-0.00665)
1. Invest in hyperparameter optimization
2. Generate 10 variants with different configs
3. Select 3 most diverse for final submissions

### If Multiple Approaches Succeed (LB < 0.00660)
1. Build ensemble of successful approaches
2. Use Bayesian optimization for ensemble weights
3. Validate ensemble diversity (r<0.80 between components)

---

## Key Insights from Research

1. **OOF predictions as features** (not blending) is a winning Kaggle pattern
2. **Smoothed target encoding** specifically designed for small data + high-cardinality
3. **Power transforms** make features Gaussian → improve Ridge regression
4. **Nested CV** eliminates information leakage → honest LB estimate
5. **TabDDPM** generates realistic synthetic data for small datasets
6. **Meta-learning** outperforms GBDTs on <1000 samples (2024 research)
7. **SHAP + RFE** prevents feature accumulation in deep stacking
8. **Uncertainty weighting** leverages calibrated prediction confidence

All 8 approaches are grounded in recent Kaggle winning solutions and peer-reviewed research (2024-2025).

---

## Conclusion

The plateau at LB 0.006698 is caused by:
1. High correlation (r>0.90) between all approaches using same feature space
2. CV-LB gap (33-81%) from information leakage in hyperparameter tuning
3. 345 samples insufficient for complex models without specialized techniques

**Breakthrough requires**:
- Different feature spaces (power transforms, OOF meta-features)
- Regularization techniques designed for small data (smoothed encoding, nested CV)
- Novel architectures (TabDDPM synthetic data, meta-learning)

**Most promising**:
1. OOF predictions as features (35% success probability)
2. SHAP meta-feature selection (30%)
3. Power transforms + ORQ (30%)

**Expected outcome**:
- 60% chance at least one approach reaches LB 0.00660-0.00665
- 25% chance of reaching LB < 0.00650
- 15% chance all approaches fail (plateau is fundamental)

**Recommendation**: Start with Tier 1 approaches (13 hours, 3 submissions). If no improvement, accept 0.006698 as near-optimal and focus on private LB diversity strategy.

---

## Sources

- [Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [OOF Meta-modelling Kaggle](https://www.kaggle.com/code/amoghjrules/and-the-stacking-continues-oof-meta-modelling)
- [Survey on Deep Tabular Learning](https://arxiv.org/abs/2410.12034)
- [Tabular Deep Learning Methods](https://arxiv.org/html/2407.00956v2)
- [Target Encoding Kaggle](https://www.kaggle.com/code/ryanholbrook/target-encoding)
- [Regularized Target Encoding](https://link.springer.com/article/10.1007/s00180-022-01207-6)
- [Power Transforms scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
- [Ordered Quantile Normalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9042069/)
- [Nested Cross-Validation scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Nested CV for Small Data](https://pubmed.ncbi.nlm.nih.gov/40039298/)
- [Bayesian Optimization Kaggle Book](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_4.xhtml)
- [Ensemble Weight Optimization](https://www.sciencedirect.com/science/article/pii/S0957417425019001)
- [Kaggle Shake-up Fundamentals](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3bc8)
- [Stacking Guide Kaggle](https://datasciblog.github.io/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
- [RocketStack Framework](https://arxiv.org/html/2506.16965)

---

**End of Report**

**Date**: 2026-02-09
**Author**: Exploration Agent
**Status**: READY FOR IMPLEMENTATION
