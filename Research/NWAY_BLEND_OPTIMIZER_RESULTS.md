# N-Way Per-Target Blend Optimizer Results

Date: 2026-02-16
Script: scripts/nway_blend_optimizer.py
Runtime: 160 seconds

## Objective

Build Frankenstein submissions by independently optimizing blends per target.
Current best: Sub 2716, LB 0.006343. Target: 0.0059.

## Methodology

### Candidate Pool
- 35 submissions loaded (10 known-LB-score anchors + 22 diverse + 3 legacy diverse)
- Diversity measured via per-target Pearson correlation with Sub 2716

### Per-Target MSE Estimation
- Estimated target MSE ratios from prediction variance across known-score subs:
  - angle: 39.7% of total MSE
  - depth: 21.7% of total MSE
  - LR: 38.6% of total MSE
- Unknown-score subs: MSE estimated with correlation-based penalty (1x-5x based on r)

### Optimization Methods
1. N-way scipy SLSQP optimizer with 20 random restarts, max 7 non-zero sources
2. Exhaustive 2-way grid search (0.01 step) over all pairs
3. 3-way grid search (0.05 step) over top 12 subs by estimated MSE

### Surrogate MSE Formula
MSE(sum_i w_i X_i) = sum_i sum_j w_i w_j rho_ij sqrt(MSE_i MSE_j)

## Key Findings

### Diversity Landscape
Submissions with highest diversity vs Sub 2716 (per target):
- angle: Sub 963 (r=-0.19), Sub 662 (r=0.07), Sub 1103 (r=-0.14)
- depth: Sub 662 (r=-0.57), Sub 653 (r=-0.22), Sub 1103 (r=0.08)
- LR: Sub 662 (r=-0.27), Sub 1103 (r=-0.05), Sub 2988 (r=0.16)

WARNING: Very diverse subs (r < 0.3) have drastically different prediction distributions
(lower variance, different means) suggesting poor quality. Use at small weights only.

### Per-Target Optimization Results

#### Angle
N-way optimizer estimated MSE: 3.848 (x1000)
- Sub 2987: w=0.290 (r=0.90 with 2716)
- Sub 2502: w=0.269 (r=0.97 with 2716)
- Sub 963: w=0.143 (r=-0.19 with 2716)
- Sub 662: w=0.118 (r=0.07 with 2716)
- Sub 653: w=0.090 (r=0.10 with 2716)
- Sub 2667: w=0.090 (r=0.00 with 2716)

Best 2-way: 0.20*Sub963 + 0.80*Sub2716, est_MSE=5.336, rho=-0.19

#### Depth
N-way optimizer estimated MSE: 1.031 (x1000)
- Sub 662: w=0.290 (r=-0.57 with 2716 - ANTI-CORRELATED)
- Sub 2716: w=0.242 (LB=0.006343)
- Sub 2987: w=0.186 (r=0.86 with 2716)
- Sub 2704: w=0.091 (r=0.84 with 2716)
- Sub 653: w=0.078 (r=-0.22 with 2716)
- Sub 963: w=0.075 (r=0.03 with 2716)
- Sub 2602: w=0.038 (r=0.67 with 2716)

Best 2-way: 0.35*Sub662 + 0.65*Sub2063, est_MSE=1.352, rho=-0.59

#### LR
N-way optimizer estimated MSE: 2.780 (x1000)
- Sub 2716: w=0.351 (LB=0.006343)
- Sub 2988: w=0.157 (r=0.16 with 2716)
- Sub 653: w=0.141 (r=0.07 with 2716)
- Sub 662: w=0.105 (r=-0.27 with 2716)
- Sub 963: w=0.103 (r=-0.14 with 2716)
- Sub 1103: w=0.093 (r=-0.05 with 2716)
- Sub 2608: w=0.052 (r=0.70 with 2716)

Best 2-way: 0.21*Sub662 + 0.79*Sub2716, est_MSE=4.843, rho=-0.27

### Known-Sub-Only Optimizer
When restricted to known-LB-score subs only:
- angle: 100% Sub 2716 (no improvement possible from known subs alone)
- depth: 100% Sub 2716
- LR: 89.7% Sub 2716 + 10.3% Sub 784 (small decorrelation from diverse LR in 784)

## Generated Submissions (54 total, Subs 3077-3130)

### Tier 1: HIGHEST PRIORITY for LB testing
These are the most likely to improve over Sub 2716:

| Sub | Strategy | Description | Risk |
|-----|----------|-------------|------|
| 3108 | cherry_pick_1 | angle=2716, depth=2503, LR=2716 | Very Low |
| 3113 | known_2way_5% | 95% 2716 + 5% best known per target (angle=2503, depth=2063, LR=784) | Very Low |
| 3114 | known_2way_10% | 90% 2716 + 10% best known per target | Low |
| 3120 | known_only_opt | Known-sub optimizer (89.7% 2716 + 10.3% 784 on LR only) | Very Low |
| 3121 | known_opt_cons | 90% 2716 + 10% known-only optimizer | Very Low |
| 3117 | hedged_top3 | Inv-MSE weighted (2716:33.8%, 2503:33.1%, 2475:33.1%) | Very Low |

### Tier 2: MODERATE PRIORITY
Mild diversity, some risk:

| Sub | Strategy | Description | Risk |
|-----|----------|-------------|------|
| 3078 | cons_90% | 90% 2716 + 10% full optimizer per target | Low |
| 3084 | perturb_5%_1507 | 95% 2716 + 5% Sub 1507 (trajectory model) | Low |
| 3092 | perturb_5%_2832 | 95% 2716 + 5% Sub 2832 (LASSO k-NN) | Low |
| 3103 | cascade_5% | 95% 2716 + 5% full optimizer output | Low |
| 3128 | surgical_LR_5% | 2716 with LR only tweaked 5% toward optimizer | Low |
| 3125 | surgical_depth_5% | 2716 with depth only tweaked 5% toward optimizer | Low |

### Tier 3: EXPLORATORY
More aggressive, higher risk:

| Sub | Strategy | Description | Risk |
|-----|----------|-------------|------|
| 3077 | pure_optimizer | Full N-way optimizer output (all 3 targets) | High |
| 3080 | best_2way_frank | angle=0.20*963+0.80*2716, depth=0.35*662+0.65*2063, LR=0.21*662+0.79*2716 | Medium |
| 3082 | lr_focused | 2716 angle+depth, LR from optimizer | Medium |
| 3083 | depth_focused | 2716 angle+LR, depth from optimizer | Medium |
| 3107 | cascade_50% | 50% 2716 + 50% optimizer | High |

## Key Recommendations

1. **Start with Tier 1** - cherry picks and known-quality blends have lowest risk
2. **Sub 3108 is the safest bet** - takes depth from Sub 2503 (which has the lowest estimated depth error among known subs) while keeping angle and LR from 2716
3. **LR has the most headroom** - Sub 784 has r=0.92 with 2716 on LR (moderate diversity) and the known-sub optimizer picks it at 10% weight
4. **Depth anti-correlation (Sub 662) is seductive but risky** - the surrogate model loves it but Sub 662 has very different prediction distribution (std=0.048 vs 0.098 for Sub 2716)
5. **If Tier 1 shows signal, escalate to Tier 2 with higher weights**

## Reproduction

```bash
uv run python scripts/nway_blend_optimizer.py
```

NOTE: Due to atomic submission numbering, re-running will produce different submission numbers. The content is deterministic given the same input submissions.
