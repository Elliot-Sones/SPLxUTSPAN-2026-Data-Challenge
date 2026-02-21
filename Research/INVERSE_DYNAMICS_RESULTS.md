# Inverse Dynamics Feature Extraction Results (2026-02-09)

## Experiment

Scripts: `scripts/inverse_dynamics_features.py` (V1), `scripts/inverse_dynamics_v2.py` (V2)

Computed joint torques, power, kinetic energy, and energy transfer features from
keypoint motion data using simplified inverse dynamics (Newton-Euler method).

## Features Computed (31 in V2, 58 in V1)

### Joint Torques (simplified: tau = I * alpha)
- tau_elbow_at_f, tau_elbow_peak, tau_shoulder_at_f, tau_shoulder_peak

### Joint Power (P = torque x angular velocity)
- P_elbow_at_f, P_elbow_peak, P_shoulder_at_f, P_shoulder_peak, P_arm_peak

### Segment Kinetic Energy (translational + rotational)
- KE_arm_at_f, KE_arm_peak, KE_forearm_at_f, KE_upper_arm_at_f
- KE_lower_peak, KE_transfer_ratio, KE_distal_proximal

### Rate of Energy Change
- dKE_arm_at_f, dKE_arm_peak

### Cumulative Energy
- E_elbow_cum, E_shoulder_cum, E_elbow_frac

### Kinetic Chain Timing
- pk_shoulder_timing, pk_elbow_timing, delay_shoulder_to_elbow
- delay_knee_to_elbow, chain_spread

### Angular Dynamics
- omega_elbow_at_f, alpha_elbow_at_f, omega_shoulder_at_f

### Impulse
- impulse_elbow, impulse_shoulder

## Anthropometric Constants Used

From Winter (2009) "Biomechanics and Motor Control of Human Movement":
- Reference body mass: 85 kg
- Upper arm: 2.8% body mass, radius of gyration 0.322 x length
- Forearm: 1.6% body mass, radius of gyration 0.303 x length
- Hand: 0.6% body mass, radius of gyration 0.297 x length
- Thigh: 10.0% body mass, radius of gyration 0.323 x length
- Shank: 4.7% body mass, radius of gyration 0.302 x length

## Feature Correlations with Targets

### Angle (several strong correlations)
| Feature | r |
|---------|---|
| KE_arm_peak | +0.7276 |
| dKE_arm_peak | +0.7157 |
| E_elbow_frac | +0.5740 |
| P_elbow_peak | +0.5094 |
| E_shoulder_cum | +0.4970 |
| chain_spread | -0.4095 |

### Depth (weak correlations - max |r| = 0.26)
| Feature | r |
|---------|---|
| omega_elbow_at_f | +0.2585 |
| KE_transfer_ratio | +0.2361 |
| E_elbow_cum | -0.1881 |

### Left_right (very weak - max |r| = 0.13)
| Feature | r |
|---------|---|
| chain_spread | -0.1169 |
| alpha_elbow_at_f | -0.1169 |
| tau_elbow_at_f | -0.1005 |

## V1 Results: Full Feature Integration (58 features)

Adding all 58 dynamics features to 213 HC+PLS features HURTS performance:
- Angle: +24.03% WORSE (LOO 0.003114 vs 0.002511 baseline)
- Depth: +11.71% WORSE (LOO 0.005039 vs 0.004510)
- Left_right: +15.92% WORSE (LOO 0.004879 vs 0.004209)
- Mean: +16.06% WORSE

Cause: curse of dimensionality. 271 features on 345 samples. Distance computation for
locally weighted regression becomes dominated by noise dimensions.

## V2 Results: Surgical Feature Selection + Prediction Blending

### Strategy A: HC+PLS baseline (matches Sub 1350)
| Target | LOO MSE |
|--------|---------|
| Angle | 0.002511 |
| Depth | 0.004510 |
| Left_right | 0.004209 |
| Mean | 0.003743 |

### Strategy B: Dynamics-only
| Target | LOO MSE |
|--------|---------|
| Angle | 0.007532 |
| Depth | 0.006391 |
| Left_right | 0.013708 |
| Mean | 0.009210 |

### Strategy C: Top-N non-redundant features added to HC+PLS
| Target | Top-3 | Top-5 | Top-8 |
|--------|-------|-------|-------|
| Angle | 0.002562 (+2.04%) | 0.002678 (+6.67%) | 0.002749 (+9.50%) |
| Depth | 0.004572 (+1.37%) | 0.004521 (+0.23%) | 0.004632 (+2.70%) |
| Left_right | 0.004287 (+1.85%) | 0.004318 (+2.59%) | 0.004294 (+2.01%) |

Feature selection hurts. Even the most "non-redundant" features add noise.

### Strategy D: Prediction-level blending
| Target | Best w | LOO MSE | Change |
|--------|--------|---------|--------|
| Angle | 0.00 | 0.002511 | 0.00% |
| Depth | 0.20 | 0.004389 | **-2.70%** |
| Left_right | 0.00 | 0.004209 | 0.00% |
| Mean | - | 0.003703 | -1.08% |

**Only depth benefits from inverse dynamics predictions** (at 20% blend weight).

### Diversity with HC-only (test predictions)
| Target | r(dynamics, HC) |
|--------|----------------|
| Angle | 0.9160 |
| Depth | 0.8392 |
| Left_right | 0.4589 |

### Correlation with Existing Submissions (best combined predictions)
| Target | r with Sub 784 | r with Sub 1350 |
|--------|----------------|-----------------|
| Angle | 0.9334 | 0.9334 |
| Depth | 0.9531 | 0.9800 |
| Left_right | 0.8386 | 0.9766 |

## Generated Submissions

### From V1 (scripts/inverse_dynamics_features.py)
| Sub | Description |
|-----|-------------|
| 1597 | Standalone combined HC+PLS+Dynamics |
| 1598 | Blend with Sub 784: aw=0.00, dw=0.30, lw=0.50 |
| 1599 | Blend with Sub 784: aw=0.00, dw=0.20, lw=0.30 |
| 1600 | Blend with Sub 784: aw=0.00, dw=0.40, lw=0.60 |
| 1601 | Blend with Sub 784: aw=0.10, dw=0.30, lw=0.50 |
| 1602 | 10% dynamics with Sub 1350 |
| 1603 | 20% dynamics with Sub 1350 |
| 1604 | 30% dynamics with Sub 1350 |
| 1605 | Dynamics-only blend with Sub 784 |

### From V2 (scripts/inverse_dynamics_v2.py)
| Sub | Description |
|-----|-------------|
| 1606 | Best per-target blend with Sub 784: aw=0.00, dw=0.30, lw=0.50 |
| 1607 | Conservative blend with Sub 784: aw=0.00, dw=0.20, lw=0.30 |
| 1608 | 5% dynamics-only blend with Sub 1350 |
| 1609 | 10% dynamics-only blend with Sub 1350 |
| 1610 | 15% dynamics-only blend with Sub 1350 |
| 1611 | Dynamics-only depth=0.20 lr=0.30 with Sub 784 |

## Key Insights

1. **Inverse dynamics features have strong correlations with angle** (KE_arm_peak r=0.73,
   dKE_arm_peak r=0.72) but these are highly redundant with existing HC features
   (max r_hc = 0.91). The arm's kinetic energy is essentially captured by arm position
   and velocity, which HC features already encode.

2. **For depth, dynamics features provide modest new signal** (omega_elbow_at_f r=0.26
   with only r=0.87 HC redundancy), enabling a small -2.70% LOO improvement via
   prediction blending at 20% weight.

3. **For left_right, inverse dynamics are nearly irrelevant** (max |r| = 0.13). Left_right
   deviation is primarily about lateral arm alignment, not about how much force/energy
   is generated.

4. **Feature-level integration ALWAYS hurts** due to curse of dimensionality.
   Prediction-level blending is the only viable approach.

5. **The fundamental limitation**: simplified inverse dynamics from noisy mocap data
   produces features that are dominated by the same position/velocity information
   already captured by HC features. True torques require accurate inertial parameters
   and ground reaction forces, which we cannot measure from keypoints alone.

## Reproduction

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge

# V1: Full feature set
uv run python scripts/inverse_dynamics_features.py
# Runtime: ~63s

# V2: Surgical selection
uv run python scripts/inverse_dynamics_v2.py
# Runtime: ~80s
```

Requires: scipy, lightgbm (for PLS), scikit-learn, joblib, pandas, numpy
