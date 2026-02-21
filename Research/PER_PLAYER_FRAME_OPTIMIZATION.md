# Per-Player Per-Target Optimal Frame Search

## Date: 2026-02-14
## Script: scripts/per_player_frame_optimization.py

## Overview

Instead of using fixed TARGET_FRAMES={angle:153, depth:150, left_right:170} for all players,
we searched for the optimal feature-extraction frame per player per target combination.

## Methodology

- Pipeline: identical to per_example_pipeline.py (locally weighted Ridge with PLS augmentation)
- Only the frame parameter varies (bandwidth_quantile=0.3, Ridge alpha=10.0)
- Frame sweep: 120 to 190 in steps of 5 (15 frames)
- LOO within each player's training data
- 5 players x 3 targets = 15 optimization runs
- Also tested robust selection (median of top-3 frames)

## Key Results

### LOO MSE Comparison

| Target     | Fixed MSE  | OptFrame MSE | Delta   |
|------------|-----------|-------------|---------|
| angle      | 0.002645  | 0.002480    | -6.25%  |
| depth      | 0.004601  | 0.004150    | -9.82%  |
| left_right | 0.004331  | 0.003372    | -22.13% |
| MEAN       | 0.003859  | 0.003334    | -13.61% |

NOTE: Fixed frame baseline was run at frames 150/150/170 (nearest to 153/150/170 in the step-5 grid).

### Optimal Per-Player Frames

**Angle** (fixed: 153):
- Player 1: 140 (MSE 0.000387) - early release
- Player 2: 155 (MSE 0.003702) - near default
- Player 3: 165 (MSE 0.001761) - late
- Player 4: 150 (MSE 0.001480) - near default
- Player 5: 150 (MSE 0.004934) - near default

**Depth** (fixed: 150):
- Player 1: 155 (MSE 0.003172, -6.94% vs fixed)
- Player 2: 140 (MSE 0.003607, -14.92%)
- Player 3: 130 (MSE 0.000652, -22.43%) - very early!
- Player 4: 180 (MSE 0.004438, -13.11%) - very late!
- Player 5: 140 (MSE 0.008511, -5.96%)

**Left_Right** (fixed: 170):
- Player 1: 185 (MSE 0.006772, -11.54%)
- Player 2: 130 (MSE 0.002255, -45.51%) - massive improvement, very early
- Player 3: 140 (MSE 0.001518, -27.79%) - early
- Player 4: 180 (MSE 0.002520, -12.35%)
- Player 5: 145 (MSE 0.003627, -23.19%) - early

### Key Observations

1. **left_right has the biggest gains**: -22.13% from per-player frames. This makes sense as lateral motion timing varies most between players.

2. **Player 2 is a massive outlier for left_right**: frame 130 vs fixed 170, a 40-frame difference giving -45.51% improvement. This player's lateral alignment happens much earlier than others.

3. **Players 3 and 4 have opposite timing for depth**: Player 3 optimal at 130 (very early, -22.43%), Player 4 at 180 (very late, -13.11%). This confirms the fixed frame was a poor compromise.

4. **Angle has the smallest improvement**: -6.25%. The fixed frame 153 was already close to optimal for most players.

5. **Robust vs optimal**: The robust selection (median of top-3) loses only 0.6% vs optimal (-12.99% vs -13.61%). Suggests the gains are real, not from single-frame noise.

### Diversity with Best Submissions

Correlations with Sub 2169 (current LB best):
- angle: r = 0.9850
- depth: r = 0.9528
- left_right: r = 0.9633

Correlations with Sub 2063:
- angle: r = 0.9739
- depth: r = 0.9336
- left_right: r = 0.9466

Moderate decorrelation, especially for depth and left_right. Blending should help.

### Overfitting Warning

Per-player optimization on ~66-74 shots is risky. The -13.61% LOO improvement is large but could be partly from fitting noise. Previous experience shows LOO improvements often translate at only 20-50% to LB. Recommend:
- Blend conservatively (10-20% weight) with known-good submissions
- The robust frame selection is safer than pure optimal

## Submissions Generated

- Sub 2371: STANDALONE per-player-frame (optimal), LOO MSE: 0.003334
- Sub 2372: 10% per-player-frame + 90% Sub 2169
- Sub 2373: 20% per-player-frame + 80% Sub 2169
- Sub 2374: 30% per-player-frame + 70% Sub 2169
- Sub 2375: 10% per-player-frame + 90% Sub 2063
- Sub 2376: 20% per-player-frame + 80% Sub 2063
- Sub 2377: 30% per-player-frame + 70% Sub 2063

## Recommended LB Testing Priority

1. **Sub 2372** (10% + 90% Sub 2169) - safest, small blend
2. **Sub 2373** (20% + 80% Sub 2169) - moderate blend
3. **Sub 2371** (standalone) - test standalone signal
4. **Sub 2375** (10% + 90% Sub 2063) - test against 2063 base

## Reproduction

```bash
uv run python scripts/per_player_frame_optimization.py
```

Total runtime: 764.9s (~12.7 minutes)
