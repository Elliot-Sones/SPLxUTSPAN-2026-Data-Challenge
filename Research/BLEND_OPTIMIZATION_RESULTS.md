# Blend Optimization Results

**Date:** 2026-02-14
**Script:** scripts/blend_optimization.py
**Runtime:** 3.2s
**Submissions generated:** 55 (Sub 2294-2348)

## Context

Current best: Sub 2169 (LB 0.006552) = 30% 3-frame ensemble + 70% Sub 2063.
Goal: Find blends that either (a) are close to Sub 2169 but slightly improved, or (b) are diverse enough to provide new signal for future ensembles.

## Key Discovery: Sub 2191 (Cauchy) is NOT the correct Cauchy standalone

Verification revealed that 10% Sub2191 + 90% Sub2063 does NOT reproduce Sub 2194 (max diff = 0.026). This means Sub 2191 was generated with a different pipeline than what was used for Sub 2194. The Cauchy component in Sub 2194 was likely generated with per-target optimized Cauchy parameters, not the single-config Sub 2191.

**Impact:** Three-way combines using Sub 2191 as "Cauchy standalone" are mixing a different Cauchy variant than what actually scored well in Sub 2194. The blends are still valid combinations, but may not reproduce the exact Cauchy signal that helped Sub 2194.

## Diversity Analysis (correlation with Sub 2169)

| Source | r_angle | r_depth | r_LR | mean_r |
|--------|---------|---------|------|--------|
| Sub 2166 (5f multiframe) | 1.0000 | 0.9995 | 0.9998 | 0.9998 |
| Sub 2194 (10% Cauchy) | 0.9991 | 0.9959 | 0.9957 | 0.9969 |
| Sub 2063 (base) | 0.9981 | 0.9950 | 0.9963 | 0.9965 |
| Sub 2152 | 0.9978 | 0.9951 | 0.9964 | 0.9964 |
| Sub 2151 | 0.9976 | 0.9952 | 0.9964 | 0.9964 |
| Sub 2191 (Cauchy standalone) | 0.9955 | 0.9899 | 0.9721 | 0.9858 |
| Sub 2163 (5f standalone) | 0.9909 | 0.9746 | 0.9885 | 0.9847 |
| **Sub 1507 (trajectory)** | **0.9840** | **0.6129** | **0.9687** | **0.8552** |

**Key finding:** Sub 1507 (trajectory distance) has dramatically different depth predictions (r=0.61 with Sub 2169). This is the most diverse signal available by far. If trajectory depth predictions are even partially correct, blending should help significantly.

## Task A: Three-Way Combine (Cauchy + Multiframe 3f + Sub2063)

The three-way combine produces predictions very close to Sub 2169 (the best known blend is already 30% 3f + 70% Sub2063). Adding Cauchy at small weights shifts slightly.

Best candidates:
- **Sub 2294**: 5% Cauchy + 30% 3f + 65% Sub2063 (closest to Sub 2169, mean_r=1.0000)
- **Sub 2295**: 10% Cauchy + 30% 3f + 60% Sub2063
- **Sub 2296**: 15% Cauchy + 30% 3f + 55% Sub2063

Also generated blends of three-way candidates WITH Sub 2169 (Sub 2299-2307). These are extremely close to Sub 2169 (mean_diff < 0.0003) - essentially micro-perturbations.

## Task B: Trajectory Distance Blends

Trajectory distance (Sub 1507) has extreme depth diversity (r=0.61). Generated:
- **Sub 2308-2313**: Uniform blends 5-30% traj + Sub 2169
- **Sub 2314-2322**: Per-target optimized (trajectory for depth only, since trajectory hurts LR)

Key per-target trajectory submissions:
- **Sub 2314**: Only 10% trajectory on depth, nothing on angle/LR
- **Sub 2315**: Only 20% trajectory on depth
- **Sub 2316**: Only 30% trajectory on depth
- **Sub 2317-2322**: Small angle weight (10-20%) + depth weight (10-30%)

## Task C: Fine-Grid Pairwise/Three-Way Blends

All top 6 LB-tested submissions (2169, 2166, 2194, 2063, 2152, 2151) are very similar (r > 0.995 with each other). The "most diverse" pairwise blends (e.g., Sub2152 + Sub2151) still have r=0.9964 with Sub 2169.

**Conclusion:** Blending among the top 6 produces nearly identical predictions. Small improvements are possible but will be hard to distinguish from noise on LB.

Close blends saved: Sub 2328-2330 (90/85/80% Sub2169 + Sub2166)
Three-way close: Sub 2334-2336

## Task D: Smart Combinations

- **Anti-2169** (Sub 2337-2340): Per-target most diverse source from top 6, blended at 5-20%
- **Equal-weight top 6** (Sub 2341): ~16.7% each
- **Inv-LB weighted** (Sub 2342): Nearly identical to equal-weight (LB scores too close)
- **Heavy-tail** (Sub 2343): 60/20/10/10 split
- **5% perturbations** (Sub 2345-2348): 95% Sub2169 + 5% of various sources

## Priority Recommendations for LB Testing

### HIGH PRIORITY (most likely to beat Sub 2169)

1. **Sub 2294** - Three-way: 5% Cauchy + 30% 3f + 65% Sub2063
   - Nearly identical to Sub 2169 but adds Cauchy signal. If the Cauchy signal in Sub 2194 helped, this should also help.

2. **Sub 2314** - Per-target trajectory: depth only at 10%
   - Adds trajectory distance diversity ONLY to depth (where r=0.61). Minimal risk since angle/LR untouched.

3. **Sub 2308** - 5% trajectory + 95% Sub2169
   - Conservative trajectory blend. Small weight minimizes downside risk.

4. **Sub 2343** - 60% Sub2169 + 20% Sub2166 + 10% Sub2194 + 10% Sub2063
   - Weighted combination of all 4 best LB subs. Acts like an ensemble of ensembles.

### MEDIUM PRIORITY (diverse, worth testing if slots available)

5. **Sub 2315** - Per-target trajectory: depth only at 20%
6. **Sub 2309** - 10% trajectory + 90% Sub2169
7. **Sub 2295** - Three-way: 10% Cauchy + 30% 3f + 60% Sub2063
8. **Sub 2337** - 5% anti-2169 + 95% Sub2169

### LOW PRIORITY (exploratory)

9. Sub 2328 - 90% Sub2169 + 10% Sub2166
10. Sub 2344 - Equal top 3
11. Sub 2341 - Equal-weight top 6
12. Sub 2317 - Per-target trajectory: 10% angle + 10% depth

## Analysis Notes

1. **Top 6 submissions are too correlated (r>0.995) for meaningful pairwise blending.** The differences are in the 4th decimal place and will be noise on LB.

2. **Trajectory distance is the only genuinely diverse signal.** Depth r=0.61 means trajectory sees fundamentally different patterns than the feature-based approach. If trajectory depth is even slightly informative, Sub 2314 (10% traj depth) is a free lunch.

3. **The Cauchy reblend mismatch** suggests Sub 2194's Cauchy component was generated with different parameters than Sub 2191. Future work should identify what exactly Sub 2194 contains.

4. **Risk profile:** Conservative blends (Sub 2294, 2308, 2314, 2343) have <0.001 mean prediction difference from Sub 2169. Even if they don't improve, they're unlikely to be worse.
