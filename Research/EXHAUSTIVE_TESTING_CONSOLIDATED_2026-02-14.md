# Exhaustive Testing - Consolidated Results (2026-02-14)

5 parallel agents tested 7 approaches. 88 submission candidates generated (Sub 2283-2377).
Runtime: ~15 minutes wall clock.

## Results Summary

| # | Approach | LOO Mean MSE | vs Baseline | Diversity (r vs 2169) | Verdict |
|---|----------|-------------|-------------|----------------------|---------|
| 1 | Target denoising (ridge_a0.5) | 0.002599 | **-30.6%** | r=0.981-0.995 (low) | LOO INFLATED - circular reasoning risk |
| 2 | Per-player frame optimization | 0.003334 | **-13.61%** | r=0.953-0.985 (moderate) | STRONGEST NEW SIGNAL |
| 3 | Mixup augmentation (alpha=1.0) | 0.003595 | **-9.34%** | r=0.977 (low) | Angle-only (-25%), overfit risk |
| 4 | Adaptive per-player bandwidth | 0.003664 | **-5.06%** | r~0.99 (very low) | Moderate, safe |
| 5 | Blend optimization (55 subs) | N/A | N/A | r=0.61 depth (trajectory!) | TRAJECTORY DIVERSITY KEY FIND |
| 6 | Three-way combine | N/A | N/A | Small perturbation | Incremental |
| 7 | Stacked residual correction | 0.003859 | **0.00%** | N/A | DEAD |

Baseline LOO: 0.003859 (bw=0.3, fixed frames, no augmentation)

## Per-Target Breakdown

| Approach | Angle LOO | Depth LOO | LR LOO |
|----------|-----------|-----------|--------|
| Baseline | 0.002645 | 0.004601 | 0.004331 |
| Target denoising | 0.002186 (-12.9%) | 0.002838 (-37.1%) | 0.002773 (-34.1%) |
| Per-player frames | 0.002480 (-6.25%) | 0.004149 (-9.82%) | 0.003373 (-22.13%) |
| Mixup | 0.001980 (-25.16%) | 0.004566 (-0.76%) | 0.004240 (-2.10%) |
| Adaptive bandwidth | 0.002539 (-4.00%) | 0.004522 (-1.72%) | 0.003930 (-9.25%) |

## Key Discoveries

### 1. Players have radically different optimal frames
- Player 2 left_right: frame 130 vs fixed 170 (40 frames apart, -45.51% MSE)
- Player 3 depth: frame 130 vs fixed 150 (-22.43%)
- Player 4 depth: frame 180 vs fixed 150 (-13.11%)
- The fixed TARGET_FRAMES were a bad compromise

### 2. Trajectory distance has extreme depth diversity
- Sub 1507 (trajectory distance) correlates only r=0.61 with Sub 2169 on depth
- All other top submissions are r>0.995 with each other
- This is the single most diverse signal available for blending

### 3. Target denoising LOO is inflated
- -30.6% LOO but uses the same LOO residuals it's denoising (circular)
- Correlation r=0.981-0.995 with Sub 2169 (very similar predictions despite LOO gain)
- Small blend weights (10%) recommended

### 4. Mixup helps angle only
- -25.16% on angle but <2% on depth/LR
- Angle has 2.57x overfit ratio - this LOO gain likely doesn't transfer

### 5. Wider bandwidth is systematically better
- Nearly all players prefer bw=0.45 (the search boundary)
- Consider extending to bw=0.50-0.60
- Or just use global bw=0.45 instead of per-player selection

### 6. Residual correction is dead
- The base model already captures all predictable signal
- Joint angles + release frame cannot predict residuals
- No alpha or damping setting improved over 0%

## LB Testing Priority Queue

### HIGH PRIORITY (most likely to beat Sub 2169)

| Priority | Sub | Description | Rationale |
|----------|-----|-------------|-----------|
| 1 | **2372** | 10% per-player frames + 90% Sub 2169 | Strongest LOO (-13.6%), moderate diversity, conservative blend |
| 2 | **2314** | Per-target: 10% trajectory depth only | Extreme depth diversity (r=0.61), untouched angle/LR |
| 3 | **2354** | 10% adaptive bandwidth + 90% Sub 2169 | Safe approach, -5.06% LOO |
| 4 | **2283** | 10% target denoising + 90% Sub 2169 | Large LOO but circular risk |
| 5 | **2365** | 10% mixup + 90% Sub 2169 | Angle-heavy gain, overfit risk |
| 6 | **2294** | 5% Cauchy + 30% 3f + 65% Sub 2063 | Three-way combine |

### MEDIUM PRIORITY

| Sub | Description |
|-----|-------------|
| 2373 | 20% per-player frames + 80% Sub 2169 |
| 2308 | 5% trajectory + 95% Sub 2169 |
| 2315 | Per-target: 20% trajectory depth only |
| 2356 | 20% adaptive bandwidth + 80% Sub 2169 |
| 2343 | 60% Sub2169 + 20% Sub2166 + 10% Sub2194 + 10% Sub2063 |

### LOW PRIORITY

| Sub | Description |
|-----|-------------|
| 2371 | Per-player frames standalone (high overfit risk) |
| 2284 | 20% target denoising + 80% Sub 2169 |
| 2366 | 20% mixup + 80% Sub 2169 |
| 2352 | Adaptive bandwidth standalone |

## Files Generated

| Agent | Subs | Script | Research |
|-------|------|--------|----------|
| denoising-agent | 2283-2293 (14) | scripts/target_denoising.py | TARGET_DENOISING_LB_CANDIDATES.md |
| blend-agent | 2294-2348 (55) | scripts/blend_optimization.py | BLEND_OPTIMIZATION_RESULTS.md |
| bandwidth-agent | 2352-2363 (12) | scripts/adaptive_bw_residual.py | ADAPTIVE_BW_RESIDUAL_RESULTS.md |
| mixup-agent | 2364-2370 (7) | scripts/mixup_focused.py | MIXUP_LB_CANDIDATES.md |
| frame-agent | 2371-2377 (7) | scripts/per_player_frame_optimization.py | PER_PLAYER_FRAME_OPTIMIZATION.md |

Total: 95 new submissions (Sub 2283-2377)

## What Didn't Work

| Approach | Why |
|----------|-----|
| Stacked residual correction | Base model already captures all signal; residuals are noise |
| Fine-grid blend of top 6 LB subs | All r>0.995 with each other (no diversity to exploit) |

## Next Steps

1. Submit HIGH PRIORITY subs to Kaggle LB (6 submissions)
2. If per-player frames work on LB: combine with adaptive bandwidth and multiframe ensemble
3. If trajectory depth diversity works: stack it with per-player frames
4. Consider extending bandwidth search to bw=0.50-0.60
5. Consider combining per-player frames + adaptive bandwidth + Cauchy kernel in one pipeline
