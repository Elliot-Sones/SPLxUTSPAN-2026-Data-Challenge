# Five Approaches Results (2026-02-12)

## Goal
Close 10.9% gap from LB 0.006619 to 0.0059 (proven achievable on LB).

## Per-Target MSE Decomposition (Sub 1828)
- Angle: ~0.006217 (overfit 2.48x, LOO 0.002511)
- Depth: ~0.006820 (overfit 1.51x)
- LR: ~0.006820 (overfit 1.62x)

## Five Approaches Tested

### Approach 1: Test-Time Augmentation (TTA)
- Method: Add Gaussian noise (sigma=0.005-0.020 feet, matching 1.4-6.6mm keypoint noise) to test keypoints, re-extract features, average 20-30 per-example Ridge predictions
- Sigma tested: 0.005, 0.008, 0.013, 0.020
- TTA prediction std: angle 0.031, depth 0.048, LR 0.038
- Diversity vs Sub 1828: r=0.95-0.98 (moderate)
- LB Results:
  - Sub 1999 (TTA only, blended with Sub 784): LB 0.006717 (WORSE, but weak base)
  - Sub 2037 (Sub 1828 + TTA depth/LR dw=0.15 lw=0.25): LB 0.006633 (flat)
  - Sub 2026 (TTA + Joint Angles): LB 0.006800 (WORSE - TTA hurts)
- **VERDICT: TTA does NOT help. Noise perturbations add variance to predictions rather than reducing it. The per-example Ridge is already smooth enough.**

### Approach 2: Per-Player Nested CV Regularization
- Method: LOPO outer loop to honestly select per-player alpha and bandwidth
- Result: All players converge to bw=0.7, alpha=5-10 (vs baseline bw=0.3, alpha=10)
- LOO MSE: 919-1652% worse than baseline (expected - wider bandwidth = less local fitting)
- Diversity vs Sub 1828: r=0.96 (most diverse of non-broken approaches)
- Not yet tested on LB as standalone
- **VERDICT: Needs LB testing. Higher LOO may mean less overfit.**

### Approach 3: Transductive PCA Distance
- Method: PCA on combined train+test features for kernel distance computation
- LOO improvement: -0.65% to -1.83% (marginal)
- Diversity vs Sub 1828: r=0.974-0.984 (nearly identical)
- **VERDICT: Essentially no effect. Distance metric barely changes.**

### Approach 4: Bayesian Ridge with Global Prior
- Method: Shrink per-example Ridge toward global model: (X'WX + alpha*I)b = X'Wy + alpha*b_prior
- LOO MSE: 9200-16600% WORSE (catastrophic)
- Diversity vs Sub 1828: r=0.51-0.59 (very diverse but garbage)
- Problem: Global prior coefficients pull local models in wrong direction. The global and local optima point differently. Coefficient-space priors don't work for this kernel regression setup.
- **VERDICT: BROKEN. Approach fundamentally flawed. Prediction-space blending (which we already do) is the correct way to shrink toward global.**

### Approach 5: Joint Angle Features
- Method: Add 10 body-proportion-invariant joint angles as supplementary features
- Features added: shoulder elevation, trunk forward/lateral lean, R/L knee flexion, wrist deviation, arm line angle, shoulder rotation, hip-shoulder twist, elbow height
- LOO improvement: angle -7.48%, depth -6.05%, LR -3.55%
- Diversity vs Sub 1828: r=0.974-0.983 (similar range but different predictions)
- Total features: 223 (213 standard + 10 joint angles)
- LB Results:
  - Sub 2020 (JA blended with Sub 784 at aw=0.5 dw=0.3 lw=0.5): LB 0.006619 (TIES best despite weaker base!)
  - Sub 2063 (50/50 avg Sub 1828 + Sub 2020): **LB 0.006603 (NEW BEST)**
  - Sub 2058 (JA blended into Sub 1828): LB 0.006621 (flat)
- **VERDICT: WINNER. Joint angles add genuine complementary signal. Simple averaging beats targeted blending.**

## LB Results Table
| Sub | Config | LB Score | Delta vs 0.006619 |
|-----|--------|----------|-------------------|
| **2063** | **50/50 avg Sub1828 + Sub2020** | **0.006603** | **-0.24%** |
| 2152 | avg w=0.40 | 0.006604 | -0.23% |
| 2151 | avg w=0.35 | 0.006605 | -0.21% |
| 2058 | JA into 1828 balanced | 0.006621 | +0.03% |
| 2037 | 1828 + TTA depth/LR | 0.006633 | +0.21% |
| 1999 | TTA only + Sub784 | 0.006717 | +1.48% |
| 2026 | TTA + JA | 0.006800 | +2.73% |

## Key Findings
1. Joint angles provide genuine new signal (especially for angle: -7.48% LOO)
2. Simple 50/50 averaging works better than targeted per-target blending
3. TTA consistently hurts or is flat - noise perturbations don't help
4. Bayesian Ridge (coefficient-space prior) is fundamentally broken for this setup
5. Transductive PCA is negligible
6. Optimal averaging weight is w=0.50 on Sub 2020 (flat curve: 0.40-0.50 all within 0.000002)

## Next Steps
1. Test nested CV predictions (most diverse: r=0.96) via similar averaging
2. Three-way average: Sub 1828 + Sub 2020 + nested_cv
3. Try joint angles with nested CV hyperparameters (bw=0.7, alpha=5)
4. Still need 0.006603 -> 0.0059 = 7.7% improvement - joint angles alone not enough

## Script
scripts/five_approaches.py
