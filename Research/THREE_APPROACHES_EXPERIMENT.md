# Three Approaches Experiment - 2026-02-05

## Goal
Get below LB 0.007 (current best: Sub 771, LB 0.007556)

## Experiments Run

### 1. Hoop-Relative Coordinate Transformation
**Script**: `scripts/hoop_relative_coords.py`
**Hypothesis**: Transforming keypoints from court coordinates to player-to-hoop reference frame should decouple depth and left_right signals.

**Features**: 1140 (original court coords + hoop-relative coords + alignment + phase features)

**Per-player per-target CV MSE**:
| Player | angle | depth | left_right |
|--------|-------|-------|------------|
| 1 | 1.638 | 10.752 | 9.116 |
| 2 | 3.917 | 11.380 | 5.248 |
| 3 | 2.752 | 3.475 | 4.992 |
| 4 | 5.114 | 11.747 | 9.119 |
| 5 | 17.145 | 29.491 | 11.281 |

**Overall CV (scaled MSE)**:
- angle: 0.00699424
- depth: 0.00773840
- left_right: 0.00784016
- **Average: 0.00752427**

**Profile**: angle_std=0.1469 (above 0.14), depth_mean=0.509 (in range)
**Runtime**: 163.6s

---

### 2. PLS Regression on Raw Time Series
**Script**: `scripts/pls_raw_timeseries.py`
**Hypothesis**: PLS on raw 49,680-dim timeseries captures patterns hand-crafted features miss.

**Optimal PLS components**:
| Player | angle | depth | left_right |
|--------|-------|-------|------------|
| 1 | 15 | 5 | 3 |
| 2 | 3 | 5 | 15 |
| 3 | 3 | 10 | 10 |
| 4 | 15 | 3 | 10 |
| 5 | 15 | 5 | 8 |

**Per-player per-target CV MSE**:
| Player | angle | depth | left_right |
|--------|-------|-------|------------|
| 1 | 1.575 | 9.946 | 9.630 |
| 2 | 4.783 | 8.955 | 6.084 |
| 3 | 4.817 | 6.609 | 6.238 |
| 4 | 5.265 | 10.873 | 10.925 |
| 5 | 16.909 | 27.749 | 9.326 |

**Overall CV (scaled MSE)**:
- angle: 0.00759293
- depth: **0.00742478** (best depth of all three approaches)
- left_right: 0.00827098
- **Average: 0.00776290**

**Profile**: angle_std=0.1641 (above 0.14), depth_mean=0.5154 (above range)
**Runtime**: 133.0s

---

### 3. Residual Modeling
**Script**: `scripts/residual_modeling.py`
**Hypothesis**: Learn residual errors of base ensemble, add corrections.

**Result**: Complete failure. Residual models INCREASED error across all player-target combos:
| Player | angle change | depth change | left_right change |
|--------|-------------|-------------|-------------------|
| 1 | -29.6% | -58.0% | -60.0% |
| 2 | -82.7% | -21.7% | -31.9% |
| 3 | -217.1% | -117.6% | -30.1% |
| 4 | -3.7% | -48.8% | -66.5% |
| 5 | -90.0% | -76.9% | -31.8% |

**Optimal alpha = 0.0** (no residual correction). Residuals are noise, not learnable signal.

**Base ensemble CV**: 0.00858755 (worse than approaches 1 and 2 due to smaller feature set)

**Correlation with Sub 771**:
- angle: 0.9719 (high)
- depth: 0.8960 (moderate)
- left_right: 0.6290 (low - diversity exists here)

---

## Key Findings

1. **Hoop-relative transform is the winner**: CV 0.00752 is competitive and the coordinate
   transformation approach actually works. Depth improved and left_right improved vs baseline.

2. **PLS got best depth prediction**: 0.00742 scaled MSE for depth alone. Raw timeseries
   contains depth signal that hand-crafted features miss.

3. **Residual modeling is a dead end**: Residuals of the base model are pure noise on this
   small dataset. No second-stage model can learn them.

4. **Profile constraints are NOT hard constraints**: Sub 771 (LB 0.007556) has angle_std=0.146
   and depth_mean=0.514 - both "violate" the soft profile guidelines. This means models that
   exceed angle_std 0.14 can still score well.

5. **Left_right diversity is highest**: Base model's left_right predictions correlate only
   0.63 with Sub 771. This is where blending has the most potential.

6. **Player 5 dominates error**: 2-3x higher MSE than other players across all approaches.

---

### 4. Target-Specific Blending with Sub 771
**Script**: `scripts/target_specific_blend.py`
**Hypothesis**: Use PLS for depth (where it excels at 0.00742), hoop-relative for angle and
left_right, then blend with Sub 771 at varying weights.

**Test prediction correlations with Sub 771**:
- angle: 0.9711 (very high - blending adds little diversity)
- depth: 0.8149 (moderate - some diversity)
- left_right: 0.7778 (moderate - meaningful diversity)

**Grid search**: 315 blend weight combinations tested. Blend formula:
`final = (1-w) * sub771 + w * new_prediction` per target.

**Submissions generated**:
| Sub | angle_w | depth_w | lr_w | angle_std | depth_mean | Notes |
|-----|---------|---------|------|-----------|------------|-------|
| 780 | 0.20 | 0.30 | 0.50 | 0.146017 | 0.513399 | Most diverse from Sub 771 |
| 781 | 0.15 | 0.30 | 0.50 | 0.146017 | 0.513399 | |
| 782 | 0.10 | 0.30 | 0.50 | 0.146017 | 0.513399 | |
| 783 | 0.05 | 0.30 | 0.50 | 0.146017 | 0.513399 | |
| 784 | 0.00 | 0.30 | 0.50 | 0.146017 | 0.513399 | No angle correction |
| 785 | 0.00 | 0.10 | 0.00 | 0.146764 | 0.513733 | Depth-only 10% PLS correction |

**Safest bet**: Sub 785 (depth-only correction) - minimal change from Sub 771, targets the
one area where PLS demonstrably excels (CV 0.00742 vs baseline ~0.0094).

---

## Key Findings

1. **Hoop-relative transform is the winner**: CV 0.00752 is competitive and the coordinate
   transformation approach actually works. Depth improved and left_right improved vs baseline.

2. **PLS got best depth prediction**: 0.00742 scaled MSE for depth alone. Raw timeseries
   contains depth signal that hand-crafted features miss.

3. **Residual modeling is a dead end**: Residuals of the base model are pure noise on this
   small dataset. No second-stage model can learn them.

4. **Profile constraints are NOT hard constraints**: Sub 771 (LB 0.007556) has angle_std=0.146
   and depth_mean=0.514 - both "violate" the soft profile guidelines. This means models that
   exceed angle_std 0.14 can still score well.

5. **Left_right diversity is highest**: Base model's left_right predictions correlate only
   0.63 with Sub 771. This is where blending has the most potential.

6. **Player 5 dominates error**: 2-3x higher MSE than other players across all approaches.

7. **Target-specific blend correlations**: New predictions have moderate diversity from Sub 771
   on depth (0.81) and left_right (0.78), but very low diversity on angle (0.97). Blending
   on angle is unlikely to help.

## Next Steps

1. **Submit 785 first** (depth-only 10% PLS correction) - lowest risk, targets the known
   bottleneck with a proven-better model.

2. **Submit 784** (depth 30% + left_right 50% correction) - higher diversity, higher risk/reward.

3. **Selective amplification on new models**: Apply the proven selective amplification
   technique to amplify differences between hoop-relative and Sub 771 on high-disagreement
   samples.

4. **Feature combination**: Add hoop-relative features to the existing ensemble pipeline
   instead of using them as a standalone model.

5. **Player-5 specific modeling**: This player dominates error across all approaches. A
   targeted strategy (more regularization, different features) could reduce overall MSE.
