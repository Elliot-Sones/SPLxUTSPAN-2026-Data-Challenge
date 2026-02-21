# Player-Targeted Optimization Results (2026-02-14)

## Premise
The 3 worst player-target combos contribute disproportionately to total error.
Instead of improving the global model, we target SPECIFIC player-target combos
with radically different approaches.

## Targets
- Player 5 depth: LOO MSE 0.008256 (baseline), ~43% of total depth error
- Player 1 LR: LOO MSE 0.008322 (baseline), ~48% of total LR error
- Player 5 angle: LOO MSE 0.004677 (baseline), ~42% of total angle error

## Sweep: 105 configs tested per combo
Bandwidth (0.10-0.95), alpha (0.1-1000), kernel (gaussian/cauchy/epanechnikov),
multi-frame (1-9 frames, spacing 3-8), cross-player transfer, feature subsets,
shrinkage, k-NN, player mean.

## Results

### Player 5 - depth (baseline 0.008256)
Top-5:
1. cross-player bw=0.80: 0.007301 (-11.57%)
2. cross-player bw=0.45: 0.007457 (-9.68%)
3. cross-player bw=0.30: 0.007545 (-8.61%)
4. bw=0.80 nf=3 sp=8: 0.008044 (-2.57%)
5. bw=0.45 nf=5 sp=3: 0.008050 (-2.49%)

KEY FINDING: Cross-player transfer (use ALL players' data with 3x upweight
for P5) is the clear winner. P5 benefits from other players' signal.

### Player 1 - left_right (baseline 0.008322)
Top-5:
1. bw=0.15: 0.006489 (-22.02%)
2. bw=0.10: 0.006527 (-21.57%)
3. bw=0.20: 0.006563 (-21.14%)
4. frame+15: 0.006728 (-19.15%)
5. bw=0.30 a=50.0: 0.006739 (-19.01%)

KEY FINDING: Very narrow bandwidth works best. P1 LR needs hyper-local
predictions - current bw=0.45 is way too wide for this combo.

### Player 5 - angle (baseline 0.004677)
Top-5:
1. bw=0.30 a=0.1: 0.001844 (-60.57%)
2. bw=0.45 a=0.1: 0.001852 (-60.40%)
3. bw=0.80 a=0.1: 0.001864 (-60.15%)
4. bw=0.80 a=1.0: 0.002017 (-56.88%)
5. bw=0.45 a=1.0: 0.002111 (-54.86%)

KEY FINDING: Very low alpha dominates. CAUTION: angle already overfits 2.57x,
reducing regularization will likely make LB worse. -60% LOO improvement is
almost certainly overfit.

## Submissions Generated
- Sub 2449: P5 depth spliced (cross-player bw=0.80)
- Sub 2450: P1 LR spliced (bw=0.15)
- Sub 2451: P5 angle spliced (bw=0.30 a=0.1) [RISKY]
- Sub 2452: ALL 3 combos spliced
- Sub 2453: ALL 3 combos 50/50 blended (SAFEST)
- Sub 2454: top-3 ensemble spliced
- Sub 2455: top-3 ensemble 50/50 blended

## Script
scripts/player_targeted_optimization.py

## What to test on LB
Priority order:
1. Sub 2453 (safest - 50/50 blend)
2. Sub 2450 (P1 LR only - cleanest signal)
3. Sub 2449 (P5 depth only - cross-player transfer)
