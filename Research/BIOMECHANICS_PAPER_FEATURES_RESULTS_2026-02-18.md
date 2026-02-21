# Biomechanics Paper Features Results - 2026-02-18

## Papers Studied

1. **Cabarkapa et al. (2023)** - "Biomechanical characteristics of proficient free-throw shooters"
   - Markerless mocap at 120Hz, 34 recreational males
   - Key differentiators (large effect sizes): knee peak angular velocity (d=1.037), COM peak velocity (d=0.988), trunk lean at release (d=0.880)
   - Proficient: lower knee/COM velocities, higher release height, backward trunk lean

2. **Li et al. (2025)** - "Arm Joint Coordination of Collegiate Basketball Athletes"
   - Vector coding for shoulder-elbow Coupled Angular Variability (CAV)
   - Higher CAV (more inter-trial variability) correlates with shooting accuracy
   - 8 coordination pattern categories based on coupling angle

3. **Chen et al. (2026)** - "Effects of playing experience on joint kinetics"
   - Joint kinetics: RTD, Peak Power, Angular Impulse for wrist/elbow/shoulder/knee
   - Wrist AI is 3.75x higher in experienced players (eta_p^2=0.54, very large)
   - VV/HV ratio: experienced players shoot more vertically

## Features Implemented

### Group A: Cabarkapa Kinematic Features (~28 features)
- COM velocity (peak, mean, at release) using ALL 69 keypoints
- COM height normalized by standing height
- Release height normalized by standing height
- Knee angular velocity profile (peak, mean over PP-RP window)
- Elbow angular velocity profile (peak, mean over PP-RP window)
- Hip angular velocity profile (peak, mean)
- Ankle angular velocity profile (peak)
- Joint ROMs (PP to RP): knee, elbow, hip, ankle
- Joint angles at release and PP: knee, elbow
- Stance width and alignment during PP
- Elbow height at PP (normalized)

### Group B: Chen Kinetics Proxies (~16 features)
- Wrist angular impulse proxy (integral of |angular velocity|)
- Wrist peak/mean angular velocity
- Elbow angular impulse proxy + peak angular velocity
- Elbow RTD proxy (peak angular acceleration)
- Shoulder angular impulse proxy + RTD proxy
- Knee peak power proxy (angular velocity * angular acceleration)
- Knee RTD proxy (peak angular acceleration)
- VV/HV decomposition: vertical velocity, horizontal velocity, ratio
- Total release speed
- Wrist peak power proxy

### Group C: Li Coupling Features (~10 features per shot)
- Coupling angle at release (last 20% of shot)
- Coupling angle at early phase (first 30%)
- Coupling angle shift (early -> release transition)
- Circular standard deviation of coupling angles
- Mean coupling angle
- Shoulder-elbow velocity ratio at release
- Elbow-wrist coupling angle + variability
- Hip-knee coupling angle

### Group D: Player-Level CAV (5 features, same per player)
- Overall CAV (cross-shot coordination variability)
- CAV in preparatory third
- CAV in loading third
- CAV in release third
- CAV phase variability (std across phases)

## Results

### LOO MSE (Leaky PLS - same framework as all our other models)
| Target | LOO MSE |
|--------|---------|
| Angle | 0.005960 |
| Depth | 0.006444 |
| Left_right | 0.008676 |
| **Mean** | **0.007027** |

### Diversity Analysis (correlation with existing submissions)
| Submission | Angle r | Depth r | LR r |
|-----------|---------|---------|------|
| Sub 2716 | 0.9824 | 0.9249 | 0.9192 |
| Sub 3294 | 0.9839 | 0.9249 | 0.9193 |
| Sub 3336 (LB best) | 0.9846 | 0.9236 | 0.9201 |

### Comparison to Other Diversity Sources
| Model | Depth r vs Sub2716 | Type |
|-------|-------------------|------|
| **Velocity CNN** | **0.66** | **Most diverse** |
| Position CNN | 0.89 | Moderate |
| **Biomech features** | **0.92** | **Moderate** |
| XGBoost tree | 0.82 | Good |
| Energy wave | 0.63 | Very diverse |
| Pulse features | 0.51 | Extremely diverse |

### Per-Player CAV Values
| Player | Overall CAV | Early | Mid | Late | Phase Std |
|--------|------------|-------|-----|------|-----------|
| P1 | 70.5 | 68.8 | 72.6 | 70.0 | 4.8 |
| P2 | 45.6 | 73.2 | 30.2 | 33.9 | 21.1 |
| P3 | 35.2 | 52.2 | 28.5 | 25.0 | 17.3 |
| P4 | 49.2 | 64.4 | 51.2 | 32.1 | 15.9 |
| P5 | 49.2 | 58.4 | 46.1 | 43.3 | 13.2 |

Note: P1 has highest and most uniform CAV (consistent variability across phases).
P3 has lowest CAV (most consistent/reproducible coordination).

## Submissions Generated
- Sub 3388: Standalone biomech features (CV=0.007027)
- Sub 3389-3392: 3/5/7/10% biomech + Sub2716
- Sub 3393-3396: 3/5/7/10% biomech + Sub3294
- Sub 3397-3400: 3/5/7/10% biomech + Sub3336 (current LB best)

## Assessment

**Strengths:**
- Biomechanically grounded features from peer-reviewed research
- CAV is a genuinely novel player-level feature
- Angular impulse proxies capture dynamics not in base features
- Moderate diversity (depth r=0.92) suggests some complementary signal

**Weaknesses:**
- Uses same Ridge pipeline, limiting diversity vs CNNs
- Diversity not as strong as velocity CNN (r=0.66 on depth)
- LOO MSE slightly worse than base pipeline
- PLS compression may be squeezing out some of the novel signal

**Best Candidates for LB Testing:**
1. Sub 3397: 3% biomech + 97% Sub3336 (conservative, low risk)
2. Sub 3398: 5% biomech + 95% Sub3336 (moderate)
3. Sub 3393: 3% biomech + 97% Sub3294 (if Sub3336 blending hurts)

**Expected Impact:**
Given diversity r~0.92 and 3-5% weight, expected improvement is small
(~0.000005-0.000015). The biomech features are most valuable as part of
a larger multi-source blend, not as the primary diversity driver.

## Script
- /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/scripts/biomechanics_paper_features.py
- Run: `uv run python scripts/biomechanics_paper_features.py --base-subs 2716 3294 3336`
