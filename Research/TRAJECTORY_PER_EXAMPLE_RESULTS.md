# Trajectory Per-Example Regression (V3) Results

## Date: 2026-02-09

## Concept
V1/V2 compute Gaussian kernel weights from distances in the 213-feature space
(extracted at a SINGLE frame). V3 instead computes distances using the FULL
240-frame trajectory of key joints, then still uses the same compact
213-feature Ridge for prediction.

Rationale: Full trajectory captures the entire shooting motion, not just a
single-frame snapshot. Two shots with identical frame-153 poses but different
wind-ups should potentially be weighted differently.

## Script
`scripts/per_example_v3.py`

## Distance Metrics Tested
1. **feat_v1**: V1-style Euclidean distance in 213-feature space (baseline)
2. **pca_traj**: PCA (20 components) on smoothed hoop-relative joint trajectories
3. **corr_traj**: Correlation distance on smoothed trajectories
4. **euc_traj**: Raw Euclidean distance on smoothed trajectories
5. **blend_X_Y**: X% PCA trajectory + Y% feature distance (normalized)

## Trajectory Configuration (Per-Target)
- **Angle**: joints=right_wrist/elbow/shoulder, neck, nose, right_hip/knee; frames 100-180
- **Depth**: joints=right_wrist/elbow/shoulder, mid_hip, right/left_hip, right/left_knee; frames 90-175
- **Left_right**: joints=right/left_wrist, right/left_shoulder, right/left_hip, right_elbow, neck; frames 110-190

## LOO CV Results

### Angle
| Method | MSE | BW | Alpha | vs V1 |
|--------|-----|-----|-------|-------|
| feat_v1 (baseline) | 0.002457 | 0.6 | 10 | - |
| pca_traj | 0.002534 | 0.6 | 10 | +3.1% worse |
| corr_traj | 0.002551 | 0.6 | 10 | +3.8% worse |
| euc_traj | 0.002533 | 0.6 | 10 | +3.1% worse |
| blend_50_50 | 0.002460 | 0.6 | 10 | +0.1% worse |
| blend_30_70 | 0.002447 | 0.6 | 10 | -0.4% better |
| blend_70_30 | 0.002484 | 0.6 | 10 | +1.1% worse |
| **blend_30_70_a5** | **0.002184** | 0.6 | **5** | **-11.1% BETTER** |
| blend_50_50_a5 | 0.002216 | 0.6 | 5 | -9.8% better |
| pca_traj_a5 | 0.002325 | 0.6 | 5 | -5.4% better |
| euc_traj_a5 | 0.002325 | 0.6 | 5 | -5.4% better |
| corr_traj_a5 | 0.002391 | 0.6 | 5 | -2.7% better |

### Depth
| Method | MSE | BW | vs V1 |
|--------|-----|-----|-------|
| feat_v1 (baseline) | 0.004473 | 0.6 | - |
| **corr_traj** | **0.004352** | 0.6 | **-2.7% BETTER** |
| euc_traj | 0.004409 | 0.6 | -1.4% better |
| pca_traj | 0.004416 | 0.6 | -1.3% better |
| blend_70_30 | 0.004414 | 0.6 | -1.3% better |
| blend_50_50 | 0.004422 | 0.6 | -1.1% better |
| blend_30_70 | 0.004437 | 0.6 | -0.8% better |

### Left_right
| Method | MSE | BW | vs V1 |
|--------|-----|-----|-------|
| **feat_v1 (baseline)** | **0.004163** | 0.6 | - |
| blend_30_70 | 0.004330 | 0.6 | +4.0% worse |
| blend_50_50 | 0.004447 | 0.6 | +6.8% worse |
| blend_70_30 | 0.004566 | 0.6 | +9.7% worse |
| corr_traj | 0.004605 | 0.6 | +10.6% worse |
| euc_traj | 0.004709 | 0.6 | +13.1% worse |
| pca_traj | 0.004728 | 0.6 | +13.6% worse |

### Overall
| Metric | Best V3 | V1 Feature | Change |
|--------|---------|-----------|--------|
| Angle | 0.002184 | 0.002457 | -11.1% |
| Depth | 0.004352 | 0.004473 | -2.7% |
| Left_right | 0.004163 | 0.004163 | 0.0% |
| **MEAN** | **0.003566** | **0.003698** | **-3.55%** |

## Diversity Analysis (Best V3 test predictions)
| Target | vs Sub 784 | vs Sub 1350 | vs V1 feat preds |
|--------|-----------|------------|------------------|
| Angle | r=0.924 | r=0.924 | r=0.999 |
| Depth | r=0.578 | r=0.610 | r=0.657 |
| Left_right | r=0.826 | r=0.971 | r=1.000 |

## Key Insights

1. **Trajectory distance helps for angle and depth, not left_right**
   - Angle benefits most from blended trajectory+feature distance (11.1% better)
   - Depth benefits from correlation distance (captures shape similarity)
   - Left_right is hurt by trajectory distance (too noisy for lateral positioning)

2. **Lower alpha (5 vs 10) matters more than distance metric for angle**
   - All trajectory distances with alpha=5 beat V1 with alpha=10
   - The alpha effect is confounded with the distance metric effect

3. **Left_right needs single-frame features, not trajectories**
   - Left_right is about body alignment at a specific moment (frame 170)
   - Full trajectory adds noise without useful signal for lateral positioning

4. **Depth diversity is notable (r=0.66 with V1 feature predictions)**
   - This suggests trajectory-based and feature-based distances find
     different similar neighbors for depth prediction

5. **LOO CV is optimistic** - these are per-example models so LOO CV
   benefits from the specific train/test structure. True LB performance
   may not show the same improvement.

## Submissions Generated
- Sub 1507: Standalone (best trajectory per target)
- Sub 1508: Blend with Sub784 (aw=0.00, dw=0.30, lw=0.50) - main candidate
- Sub 1509: Conservative (aw=0.00, dw=0.25, lw=0.40)
- Sub 1510: Aggressive (aw=0.00, dw=0.35, lw=0.55)
- Sub 1511: More conservative (aw=0.00, dw=0.20, lw=0.30)
- Sub 1512-1514: V3+V1 ensemble blends (v3_w=0.3/0.5/0.7)
- Sub 1515-1517: Blends with Sub 1350 (20%/30%/50% V3)

## Recommended Submissions to Test on LB
1. **Sub 1508** (aw=0.00, dw=0.30, lw=0.50) - best blend weights for trajectory approach
2. **Sub 1515** (80% Sub1350 + 20% V3) - conservative blend with current best
3. **Sub 1513** (V3+V1 50/50 ensemble, dw=0.30, lw=0.50) - diversified ensemble

## Warning
The angle "improvement" from alpha=5 + blended distance needs validation.
The LOO CV improvement could be overfitting to the leave-one-out evaluation.
For left_right, V3 strictly falls back to V1 (feat_v1 was best), so Sub 1508
depth predictions are the only novel component vs V1.
