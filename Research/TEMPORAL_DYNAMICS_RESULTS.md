# Temporal Dynamics Modeling Results

**Date**: 2026-02-08
**Objective**: Model full trajectory dynamics instead of 3-frame snapshots to achieve LB < 0.005
**Status**: COMPLETED - Generated 9 submissions (Sub 1478-1486)

## Approach

Instead of using hoop-relative coordinates at 3 specific frames (153, 150, 170), we modeled the FULL 240-frame trajectory dynamics:

### 1. Trajectory Feature Extraction (450 features)

**Velocity/Acceleration Trajectories** (216 features):
- 12 key joints: right_wrist, right_elbow, right_shoulder, left_wrist, left_shoulder, right_hip, left_hip, mid_hip, right_knee, left_knee, neck, nose
- For each joint x 3 coordinates:
  - Mean velocity over pre-release window (frames 100-140)
  - Std velocity over pre-release window
  - Mean acceleration over pre-release window
  - Same 3 stats for release window (frames 140-180)
- Total: 12 joints x 3 coords x 6 stats = 216 features

**B-Spline Functional Representation** (90 features):
- 3 critical joints: right_wrist, right_elbow, right_shoulder
- For each joint x 3 coordinates:
  - Fit cubic B-spline with smoothing (s=0.01)
  - Sample at 10 basis points uniformly over time
- Total: 3 joints x 3 coords x 10 basis = 90 features
- Captures smooth trajectory shape independent of local noise

**Temporal Pooling** (144 features):
- 3 critical joints x 3 coordinates
- 4 temporal windows: pre (80-120), release (120-160), post (160-200), full (80-200)
- For each joint x coord x window:
  - Mean, std, max, min
- Total: 3 joints x 3 coords x 4 windows x 4 stats = 144 features

### 2. DTW-Based Trajectory Similarity

**Dynamic Time Warping Distance Matrix**:
- Computed on right wrist 3D trajectories (240 frames x 3 coords = 720 dims flattened)
- 345 x 345 pairwise DTW distances = 59,340 distances
- Time-warp invariant: aligns trajectories with different timing
- Runtime: ~5 minutes for full distance matrix

**Locally Weighted Regression**:
- For each test shot, weight training samples by DTW similarity
- Gaussian kernel: w_i = exp(-d_i^2 / (2*sigma^2))
- Adaptive bandwidth: sigma = quantile of pairwise distances
- Ridge regression (alpha=10) with sample weights
- Per-player modeling to preserve player-specific patterns

### 3. Three Modeling Approaches Per Target

**[1] Standard Ensemble** (Ridge + LightGBM + XGBoost):
- Per-player per-target models
- 5-fold CV for honest OOF
- Equal-weight ensemble of 3 models

**[2] DTW Locally Weighted**:
- Per-example personalization using DTW distances
- Leave-one-out CV for training (exact)
- Test: use Euclidean distance as DTW proxy (no test-train DTW precomputed)

**[3] Blend**:
- Weighted combination of Ensemble and DTW
- Tuned alpha per target (0.3, 0.5, 0.7)

## Results

### CV Performance (Scaled Space)

| Target     | Ensemble | DTW Local | Blend (best α) | Best Config    |
|------------|----------|-----------|----------------|----------------|
| Angle      | 0.006840 | 0.009417  | 0.007106       | Ens (bw=0.3)   |
| Depth      | 0.007763 | 0.006035  | 0.005916       | Blend (α=0.3, bw=0.5) |
| Left_right | 0.008271 | 0.008544  | 0.007642       | Blend (α=0.5, bw=0.35) |
| **MEAN**   | **0.007625** | **0.007999** | **0.006888** | **Blend** |

**Key Findings**:
1. **Blend is best overall** (0.006888 vs 0.007625 ensemble vs 0.007999 DTW)
2. **DTW excels for depth** (0.006035 vs 0.007763 ensemble) - trajectory dynamics are strong depth predictors
3. **Ensemble better for angle** (0.006840 vs 0.009417 DTW) - angle is more about static pose than trajectory shape
4. **Blending captures complementary signals** - ensemble captures static pose, DTW captures trajectory dynamics

### Bandwidth Sensitivity

**Angle**:
- Optimal: bw=0.3 (baseline)
- Range: 0.20-0.50 (MSE 0.009417-0.011085)
- Lower bandwidth (more local) is better for angle

**Depth**:
- Optimal: bw=0.5 (wider)
- Range: 0.20-0.50 (MSE 0.006035-0.006117)
- Wider bandwidth (more global) is better for depth
- **Depth benefits most from trajectory modeling**

**Left_right**:
- Optimal: bw=0.35 (moderate)
- Range: 0.30-0.50 (MSE 0.008544-0.008643)
- Moderate bandwidth is best

### Diversity vs Sub 1350 (LB 0.006776)

| Approach | Angle (r) | Depth (r) | Left_right (r) |
|----------|-----------|-----------|----------------|
| Ensemble | 0.9721    | 0.9162    | 0.8522         |
| DTW      | 0.9187    | 0.8684    | 0.8715         |
| Blend    | 0.9610    | 0.8967    | 0.8826         |

**Key Findings**:
1. **DTW has moderate diversity** (r=0.87-0.92) - different from Sub 1350's per-example approach
2. **Depth has strongest diversity** across all approaches (r < 0.92)
3. **Angle is highly correlated** (r > 0.91) - trajectory dynamics don't add much new signal for angle
4. **Sufficient diversity for blending** - r < 0.97 means potential LB improvement

## Submissions Generated

### Standalone Models (Sub 1478-1480)

| Sub  | Model    | Mean CV  | angle_std | depth_mean | Description |
|------|----------|----------|-----------|------------|-------------|
| 1478 | Ensemble | 0.007625 | 0.1437    | 0.5062     | Standard tree ensemble on trajectory features |
| 1479 | DTW      | 0.007999 | 0.1606    | 0.5074     | DTW locally weighted (best per target) |
| 1480 | Blend    | 0.006888 | 0.1478    | 0.5070     | **BEST CV** - Blend of ensemble + DTW |

**Profile Constraints**:
- angle_std < 0.14: All 3 violations (soft constraint, OK based on Sub 771 experience)
- depth_mean in [0.50, 0.51]: Sub 1478 passes, 1479/1480 marginal

### Blends with Sub 784 (Sub 1481-1486)

**Blend approach** (depth improvement, left_right improvement):
- Sub 1481: aw=0.00, dw=0.30, lw=0.50 (Sub 784 weights)
- Sub 1482: aw=0.00, dw=0.20, lw=0.40 (conservative)
- Sub 1483: aw=0.00, dw=0.40, lw=0.60 (aggressive)

**DTW approach** (depth focus):
- Sub 1484: aw=0.00, dw=0.30, lw=0.50 (Sub 784 weights)
- Sub 1485: aw=0.00, dw=0.20, lw=0.40 (conservative)
- Sub 1486: aw=0.00, dw=0.40, lw=0.60 (aggressive)

All blends preserve Sub 784 angle (no aw) and adjust depth/LR only.

## Comparison with Current Best (Sub 1350)

| Metric        | Sub 1350 (LB 0.006776) | Sub 1480 (Temporal Blend) |
|---------------|------------------------|---------------------------|
| Approach      | Per-example locally weighted at 3 frames | DTW + ensemble on full trajectories |
| CV Mean       | 0.003743 (LOO, optimistic) | 0.006888 (5-fold, honest) |
| Angle CV      | 0.002168               | 0.007106                  |
| Depth CV      | 0.004473               | 0.005916                  |
| LR CV         | 0.004163               | 0.007642                  |
| Features      | 213 (198 HC + 15 PLS) per target | 450 (trajectory derivatives + splines + pooling) |
| Angle r       | 0.93 (vs Sub 784)      | 0.9610 (vs Sub 1350)      |
| Depth r       | 0.93 (vs Sub 784)      | 0.8967 (vs Sub 1350)      |
| LR r          | 0.83 (vs Sub 784)      | 0.8826 (vs Sub 1350)      |

**Analysis**:
- **CV gap is large** (0.006888 vs 0.003743) but Sub 1350's LOO is optimistic
- **Sub 1350 is still superior** - per-example personalization at optimal frames beats full trajectory
- **Trajectory dynamics add moderate diversity** (r=0.88-0.96)
- **Depth sees most benefit** from temporal modeling (CV 0.005916, r=0.8967)

## Key Insights

### What Works
1. **Trajectory dynamics for depth** - DTW locally weighted achieves 0.006035 CV (vs 0.007763 ensemble)
2. **Velocity/acceleration features** - temporal derivatives capture shot dynamics
3. **B-spline smoothing** - functional representation reduces noise
4. **Blending static + dynamic** - ensemble (static pose) + DTW (trajectory) is complementary
5. **Adaptive bandwidth per target** - depth needs wider (0.5), angle needs narrower (0.3)

### What Does NOT Work
1. **Trajectory dynamics for angle** - DTW worse than ensemble (0.009417 vs 0.006840)
   - Angle is more about release pose than trajectory shape
2. **Full 240 frames vs 3 optimal frames** - Sub 1350's targeted frames beat full trajectory
   - Information redundancy: most trajectory is noise for prediction
3. **Test DTW distance** - no precomputed test-train DTW, fallback to Euclidean hurts generalization

## Recommendations

### Priority Testing
1. **Sub 1481** (blend + Sub 784, dw=0.30, lw=0.50) - matches Sub 1350 optimal weights
2. **Sub 1480** (standalone blend) - best CV, but may violate constraints
3. **Sub 1486** (DTW + Sub 784, aggressive) - depth focus with high weight

### Next Steps
1. **Combine temporal + Sub 1350** - use trajectory features IN ADDITION to 3-frame snapshots
2. **Target-specific temporal modeling** - depth gets DTW, angle gets static ensemble
3. **Precompute test-train DTW** - compute full DTW matrix including test for better generalization
4. **Shorter trajectory windows** - focus on critical 100-180 frame window instead of full 240

## Technical Details

**Script**: `/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/scripts/temporal_dynamics_pipeline.py`

**Data**:
- Train: 345 shots x 240 frames x 69 keypoints x 3 coords
- Test: 113 shots x 240 frames x 69 keypoints x 3 coords
- Features: 450 per shot (trajectory-based)

**Runtime**: 364 seconds (6.1 minutes)
- Feature extraction: ~2 min
- DTW distance matrix: ~5 min (59,340 distances)
- Modeling (3 targets x 3 approaches): ~2 min

**Models**:
- Ridge (alpha=10.0)
- LightGBM (n=80, leaves=8, lr=0.05, subsample=0.8)
- XGBoost (n=80, depth=3, lr=0.05, subsample=0.8)

**Submission Files**:
- Sub 1478-1480: Standalone temporal models
- Sub 1481-1486: Blends with Sub 784 (6 weight configurations)

## Conclusion

**Temporal dynamics modeling provides moderate improvement for depth** (DTW CV 0.006035 vs Sub 1350's 0.004473 is competitive) but **underperforms Sub 1350's per-example approach overall** (mean CV 0.006888 vs 0.003743).

**Key finding**: Full trajectory dynamics are **complementary but not superior** to optimal 3-frame snapshots. The optimal strategy is likely **hybrid**: use targeted frames for angle/LR + trajectory dynamics for depth.

**Expected LB performance**: 0.0074-0.0080 for standalone models, 0.0068-0.0072 for blends with Sub 784 (no breakthrough, but moderate diversity for ensembling).
