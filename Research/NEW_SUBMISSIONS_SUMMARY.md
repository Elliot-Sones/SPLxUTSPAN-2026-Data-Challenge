# New Submissions Summary (2026-01-31)

## BREAKTHROUGH: Submissions with Lower angle_std Than Sub 133

| Sub | angle_std | Improvement | Depth Mean | Corr w/ 133 | Method |
|-----|-----------|-------------|------------|-------------|--------|
| **183** | 0.136569 | **0.40%** | 0.5055 | 0.9990 | Selective amplification |
| **184** | 0.136799 | **0.23%** | 0.5055 | 0.9998 | Combined strategies |
| **185** | 0.137105 | **0.009%** | 0.5055 | 1.0000 | w10=48% blend |

These submissions have:
- Lower angle_std than Sub 133 (0.137117)
- Optimal depth_mean (0.5055)
- High correlation with Sub 133 (good stability)

### Recommended Submission Order:
1. **Sub 183** - Best angle_std improvement (0.40%)
2. **Sub 184** - Second best improvement (0.23%)
3. **Sub 182** - w10=48% component blend

---

## Best Candidate Submissions

Based on our experiments, here are the most promising submissions to try:

### Top Priority: Best Predicted LB

| Sub | angle_std | depth_mean | Corr w/ 133 | Predicted LB | Notes |
|-----|-----------|------------|-------------|--------------|-------|
| 169 | 0.137677 | 0.5055 | 0.9999 | 0.007803 | LB-optimized per-target weights |
| 164 | 0.137674 | 0.5055 | 0.9803 | 0.007804 | Profile-constrained blend |
| 171 | 0.137627 | 0.5055 | 0.9997 | 0.007811 | 2% Sub163 diversity |

### Lower angle_std Than Sub 133

| Sub | angle_std | Improvement | Corr w/ 133 | Notes |
|-----|-----------|-------------|-------------|-------|
| 174 | 0.137065 | 0.04% | 0.9996 | 5-way blend (30% Sub133 + components) |
| 170 | 0.137069 | 0.03% | 0.9998 | 4-way blend (20% S9 + 55% S10 + 25% S111) |
| 176 | 0.137087 | 0.02% | 1.0000 | Ultra-regularized blend |
| 177 | 0.137092 | 0.02% | 1.0000 | Ensemble of regularizations |
| 180 | 0.137101 | 0.01% | 1.0000 | Fine-grained weight optimization |
| 179 | 0.132221 | 3.57% | 0.9998 | Shrinkage approach (RISKY) |

### Most Diverse (but still valid profile)

| Sub | angle_std | Corr w/ 133 | Notes |
|-----|-----------|-------------|-------|
| 164 | 0.137674 | 0.9803 | Profile-constrained blend |
| 166 | 0.1499 | 0.9748 | 33% Sub163 + 67% Sub133 (angle_std slightly high) |

## Detailed Submission Descriptions

### Sub 169 (Recommended First)
- **Model**: LB-optimized per-target blend
- **angle weight**: 2% Sub163 + 98% Sub133
- **depth weight**: 0% Sub163 + 100% Sub133
- **lr weight**: 0% Sub163 + 100% Sub133
- **Profile**: angle_std=0.1377, depth_mean=0.5055
- **Why it might work**: Optimized specifically for the LB prediction model

### Sub 170 (Recommended Second)
- **Model**: Aggressive 4-way blend (no Sub25)
- **Weights**: 0% Sub25 + 20% Sub9 + 55% Sub10 + 25% Sub111
- **Profile**: angle_std=0.1371, depth_mean=0.5055
- **Why it might work**: Slightly lower angle_std than Sub 133, different blend weights

### Sub 174 (Recommended Third)
- **Model**: 5-way blend including Sub133
- **Weights**: 30% Sub133 + 6% Sub25 + 2% Sub9 + 40% Sub10 + 22% Sub111
- **Profile**: angle_std=0.1371, depth_mean=0.5055
- **Why it might work**: Lowest angle_std found, good depth calibration

### Sub 179 (High Risk, High Reward)
- **Model**: Shrinkage of extreme predictions toward mean
- **Profile**: angle_std=0.1322 (3.57% lower than Sub 133)
- **Risk**: Shrinkage may hurt prediction accuracy
- **Why it might work**: Much lower variance could offset accuracy loss

## Scripts Created

| Script | Purpose |
|--------|---------|
| per_player_feature_analysis.py | Analyze optimal features per player |
| per_player_specialized.py | Build per-player specialized models |
| profile_constrained_blend.py | Blend with profile constraints |
| targeted_diversity_blend.py | Find optimal diversity blend |
| analyze_sub133_vs_sub151.py | Compare Sub 133 vs Sub 151 |
| per_target_blend.py | Different blend weights per target |
| outlier_handling.py | Special handling for outlier samples |
| aggressive_ensemble.py | Grid search for better blend weights |
| ultra_regularized.py | High regularization for low variance |
| micro_optimize.py | Micro-adjustments to Sub 133 |
| analyze_best_candidates.py | Rank and compare candidates |

## Key Insights

1. **Sample 17 is the #1 outlier** - Confirmed by Isolation Forest
2. **Per-player features have <10% overlap** - Different features for each player
3. **Diversity vs Profile tradeoff is real** - Cannot have both
4. **Sub 133's exact weights matter** - Tiny changes in blend weights affect LB
5. **Shrinkage can lower angle_std** - But may hurt accuracy

## Submission Priority

1. **Sub 169** - Best predicted LB (0.007803)
2. **Sub 170** - Lower angle_std with good diversity
3. **Sub 174** - Lowest angle_std found
4. **Sub 179** - High risk shrinkage approach
5. **Sub 166** - Most diverse (but angle_std slightly high at 0.15)
