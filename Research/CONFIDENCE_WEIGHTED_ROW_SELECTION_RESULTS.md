# Confidence-Weighted Row-Level Model Selection Results

**Date**: 2026-02-16
**Script**: scripts/confidence_weighted_selection.py
**Base**: Sub 2716 (LB 0.006343)

## Motivation

The Ridge model is likely very good for MOST test examples but BAD for a few hard ones.
If we identify which test examples are "hard" for Ridge and substitute a different model's
prediction for just those rows, we can improve without the dilution of global blending.

## Approach

1. Load predictions from 9 diverse models (Sub 2503, 2716, 2887, 2784, 2780, 1557, 1507, 2602, 2608)
2. For each test example, compute "confidence score" = how close it is to training data in feature space
3. Compute "diverse consensus" = median of diverse models (excluding base)
4. For low-confidence rows: blend with diverse consensus
5. For high-confidence rows: keep base (Sub 2716) unchanged

## Model Diversity vs Sub 2716

Pearson r between each model and Sub 2716:

| Model | scaled_angle r | scaled_depth r | scaled_left_right r | RMSD angle | RMSD depth | RMSD LR |
|-------|-----------|-----------|-----------|------|------|------|
| ridge_base (2503) | 0.9994 | 0.9975 | 0.9981 | 0.006 | 0.008 | 0.006 |
| bigru_10seed (2887) | 0.9468 | 0.5298 | 0.7200 | 0.158 | 0.098 | 0.083 |
| knn_bootstrap (2784) | 0.9449 | 0.7835 | 0.5011 | 0.159 | 0.074 | 0.096 |
| rf_velocity (2780) | 0.9494 | 0.7344 | 0.6992 | 0.154 | 0.086 | 0.083 |
| temporal_cnn (1557) | 0.8875 | 0.7417 | 0.7247 | 0.070 | 0.067 | 0.064 |
| trajectory (1507) | 0.9891 | 0.6106 | 0.9535 | 0.025 | 0.147 | 0.037 |
| energy_wave (2602) | 0.9279 | 0.6725 | 0.7248 | 0.067 | 0.105 | 0.078 |
| pulse (2608) | 0.8015 | 0.5330 | 0.7046 | 0.110 | 0.113 | 0.103 |

**Key finding**: Depth has the most diverse model predictions (r=0.53-0.78), followed by LR (r=0.50-0.95).
Angle is most correlated across models (r=0.80-0.99).

## Confidence Score Statistics

Confidence = 1 - percentile of test-to-train distance in the train-to-train distance distribution.
Higher = more typical/confident, lower = more atypical/hard.

| Target | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| angle | 0.548 | 0.308 | 0.000 | 1.000 |
| depth | 0.541 | 0.309 | 0.000 | 1.000 |
| left_right | 0.519 | 0.309 | 0.000 | 1.000 |

Per-player (n_below_0.3 = hard examples):
- P1: 7-8 hard examples out of 23
- P2: 2-4 hard examples out of 22 (most typical)
- P3: 4-10 hard examples out of 22
- P4: 3-7 hard examples out of 22
- P5: 6-10 hard examples out of 24 (most atypical)

## Hardest Test Examples

Consistent hardest rows across targets:
- **idx=105, P5**: conf=0.000 across all 3 targets, huge prediction spread (0.74 angle, 1.18 depth, 0.39 LR)
- **idx=65, P3**: conf=0.000 for angle/depth, 0.015 for LR
- **idx=111, P5**: conf=0.014 angle, 0.000 depth, 0.000 LR
- **idx=106, P5**: conf=0.014 angle/depth, 0.000 LR

P5 dominates the hardest rows - consistent with prior findings that P5 is hardest to model.

## Ensemble Disagreement

| Target | Mean Std | Max Std | p90 Std |
|--------|----------|---------|---------|
| scaled_angle | 0.084 | 0.199 | 0.116 |
| scaled_depth | 0.063 | 0.458 | 0.089 |
| scaled_left_right | 0.059 | 0.160 | 0.093 |

**Correlation between OOD confidence and disagreement**:
- angle: r=-0.24 (weak)
- depth: r=-0.43 (moderate)
- left_right: r=-0.36 (moderate)

Confidence and disagreement capture partially different signals.

## Generated Submissions

All 17 submissions are modifications of Sub 2716 (LB 0.006343).

| Sub | Config | Rows Changed (>0.001) | Mean RMSD vs Base |
|-----|--------|----------------------|-------------------|
| 2967 | conservative_10pct_10w | 9/11/8 | 0.002323 |
| 2968 | conservative_15pct_15w | 16/15/14 | 0.003855 |
| 2969 | moderate_20pct_15w | - | 0.004077 |
| 2970 | moderate_20pct_20w | - | 0.005436 |
| 2971 | aggressive_30pct_20w | - | 0.005753 |
| 2972 | per_target_15pct_10w | 13/15/12 | 0.002535 |
| 2973 | per_target_20pct_15w | - | 0.003997 |
| 2974 | continuous_blend_max15 | all | 0.004442 |
| 2975 | continuous_blend_max20 | all | 0.005922 |
| 2976 | per_player_target_10pct_15w | - | 0.003787 |
| 2977 | disagreement_top15pct_15w | 16/15/13 | 0.004179 |
| 2978 | combined_conf_disagree_15w | - | 0.004813 |
| 2979 | median_closest_per_row | all | 0.031253 |
| 2980 | confidence_weighted_avg | all | 0.038793 |
| 2981 | hardest_swap_w0.20 | 5/4/5 | 0.004808 |
| 2982 | hardest_swap_w0.35 | 5/4/5 | 0.008414 |
| 2983 | hardest_swap_w0.50 | 5/4/5 | 0.011734 |

## Recommended LB Test Priority

1. **Sub 2967** (conservative_10pct_10w) - LOWEST RISK: Only changes ~10 rows, minimal perturbation
2. **Sub 2981** (hardest_swap_w0.20) - MOST SURGICAL: Only swaps 5 hardest rows per player-target
3. **Sub 2972** (per_target_15pct_10w) - Independent thresholds per target
4. **Sub 2977** (disagreement_top15pct_15w) - Uses model disagreement instead of OOD distance

## Notes

- Kaggle submission API returned 400 error (likely daily limit hit). Submissions need to be uploaded
  when quota resets.
- The diverse consensus median is based on 7 non-base models with genuinely different architectures.
- Sub 2979 (median-closest) and Sub 2980 (confidence-weighted-avg) are too aggressive - RMSD > 0.03 vs base.
  These will almost certainly HURT. Only the conservative variants have a realistic chance of helping.
- The moderate negative correlation between OOD confidence and disagreement (r = -0.24 to -0.43)
  confirms these are partially independent signals for identifying hard rows.
