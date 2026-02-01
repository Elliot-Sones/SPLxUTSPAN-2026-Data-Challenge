# Path to 0.007 - Research Findings

## Executive Summary

**Current Best**: Sub 133 with LB 0.007809
**Target**: LB 0.007 (10.4% improvement needed)
**Status**: Reached practical limits of current approach

---

## Key Discovery: Per-Player Feature Overlap is Extremely Low

### Analysis Results (per_player_feature_analysis.py)

| Target | Avg Jaccard Overlap | Interpretation |
|--------|---------------------|----------------|
| angle | 0.083 | Very low - different features for each player |
| depth | 0.045 | Extremely low - almost no overlap |
| left_right | 0.065 | Very low - specialized features per player |

**Critical Finding**: Zero features appear in ALL players' top 20 for any target.

### Per-Player Top Features (angle)

| Player | Top Feature | Importance |
|--------|-------------|------------|
| Player 1 | phase_setup_mid_hip_vel_mean | 0.182 |
| Player 2 | phase_propulsion_mid_hip_vel_max | 0.326 |
| Player 3 | neck_z_max | 0.435 |
| Player 4 | phase_setup_mid_hip_z_range | 0.342 |
| Player 5 | left_wrist_z_max | 1.434 |

Each player has 10-15 unique features not in any other player's top 20.

---

## Per-Player Specialized Model Results (Sub 163)

### Model Details
- Per-player Ridge regression with player-specific top-50 features
- Separate feature sets per target per player

### Results
| Metric | Value |
|--------|-------|
| Correlation with Sub 133 (angle) | 0.8515 |
| Correlation with Sub 133 (depth) | 0.7343 |
| Correlation with Sub 133 (lr) | 0.4501 |
| angle_std | 0.1931 (target: < 0.14) |
| depth_mean | 0.4998 (target: 0.50-0.51) |

### Problem: Profile Constraint Conflict

The per-player specialized model achieves meaningful diversity (85% vs 99%+ correlation)
but violates profile constraints (angle_std too high at 0.19 vs target 0.14).

---

## Profile-Constrained Blending Results

### Weight Scan (Sub 163 blended with Sub 133)

| Weight on Sub163 | angle_std | depth_mean | Correlation |
|------------------|-----------|------------|-------------|
| 0.1 | 0.1402 | 0.5049 | 0.9974 |
| 0.2 | 0.1440 | 0.5043 | 0.9901 |
| 0.3 | 0.1484 | 0.5038 | 0.9788 |
| 0.4 | 0.1535 | 0.5032 | 0.9646 |
| 0.5 | 0.1590 | 0.5026 | 0.9480 |

**Maximum weight satisfying angle_std < 0.14: 9.4%**

This results in 99.77% correlation with Sub 133 - essentially no diversity.

---

## Sub 133 vs Sub 151 Analysis

Sub 151 (LB 0.008305) was 6.4% worse than Sub 133 (LB 0.007809) despite:
- 99.85% correlation
- Profile distance 0.0003 (nearly perfect match)

### Critical Finding: Sample 17

| Metric | Sub 133 | Sub 151 | Difference |
|--------|---------|---------|------------|
| angle | 0.6163 | 0.6447 | -0.0284 |
| depth | 0.4154 | 0.2921 | 0.1233 |
| lr | 0.3714 | 0.3379 | 0.0335 |
| Total diff | - | - | 0.1309 |

Sample 17 accounts for 0.13 of the 0.017 average difference - it's a massive outlier.

### Sub 151 = Sub 25
Sub 25 has 100% correlation with Sub 151. The "optimized blend" converged to Sub 25.

---

## Submissions Created

| Sub # | Model | angle_std | depth_mean | Corr w/ 133 | Notes |
|-------|-------|-----------|------------|-------------|-------|
| 163 | Per-player specialized | 0.1931 | 0.4998 | 0.8515 | Profile fails |
| 164 | Profile-constrained blend | 0.1377 | 0.5055 | 0.9803 | Too similar |
| 165 | 9.4% Sub163 + 90.6% Sub133 | 0.1400 | 0.5049 | 0.9977 | Too similar |
| 166 | 33% Sub163 + 67% Sub133 | 0.1499 | 0.5036 | 0.9748 | Best diversity with relaxed constraints |

---

## The Fundamental Tradeoff

```
Diversity <---> Profile Constraints

High Diversity (85% corr)  =>  Bad Profile (angle_std = 0.19)
Good Profile (angle_std < 0.14)  =>  Low Diversity (98%+ corr)
```

This tradeoff appears insurmountable with current features.

---

## Recommendations

### Option 1: Submit Sub 166 (Aggressive Blend)
- 33% per-player specialized + 67% Sub 133
- angle_std = 0.1499 (slightly above 0.14 target)
- 97.5% correlation with Sub 133
- Risk: Profile violation may hurt LB

### Option 2: Accept Sub 133 as Optimal
- Evidence strongly suggests Sub 133 is near the signal limit
- All attempts at "improvement" result in either:
  - Same predictions (99%+ correlation)
  - Worse profile (angle_std > 0.14)
- 10% improvement may not be achievable with current data

### Option 3: Focus on Sample 17
- Sample 17 is a massive outlier that differentiates Sub 133 from Sub 151
- Better handling of this single sample could be worth 0.0005 LB
- Requires understanding why Sub 133's blend handles it better

---

## Files Created

| File | Purpose |
|------|---------|
| scripts/per_player_feature_analysis.py | Analyze optimal features per player |
| scripts/per_player_specialized.py | Build specialized models |
| scripts/profile_constrained_blend.py | Blend with profile constraints |
| scripts/targeted_diversity_blend.py | Find optimal diversity blend |
| scripts/analyze_sub133_vs_sub151.py | Compare best vs near-best |
| output/per_player_top_features.csv | Combined feature importance |
| output/player*_*_feature_importance.csv | Per-player feature rankings |
| output/sub133_vs_sub151_analysis.csv | Detailed submission comparison |

---

## Conclusion

The path to LB 0.007 requires fundamentally different predictions that:
1. Maintain angle_std < 0.14
2. Maintain depth_mean in [0.50, 0.51]
3. Are meaningfully different from Sub 133 (<95% corr)

Current evidence suggests this combination may not exist with the available features.
The per-player feature analysis shows extremely low overlap (4-8% Jaccard),
confirming that different features work for different players, but exploiting
this insight while maintaining profile constraints has not been achieved.

**Final Assessment**: Sub 133 likely represents near-optimal performance.
Further improvement would require new data sources or modeling breakthroughs
not achieved in 100+ experiments.
