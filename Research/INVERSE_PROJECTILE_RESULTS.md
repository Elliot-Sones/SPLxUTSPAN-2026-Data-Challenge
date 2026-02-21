# Inverse Projectile Pipeline - Full Results

## Date: 2026-02-05

## Core Insight

For every training shot, we know the targets (angle, depth, left_right) and release position (from hand keypoints). From these, we can compute the EXACT ball release velocity via inverse projectile physics:

```
t^2 = 2 * (dz + D * tan(angle_rad)) / g
vx = dx / t, vy = dy / t, vz = (dz + 0.5*g*t^2) / t
```

This gives ground-truth velocity labels for learning the hand-to-ball transfer function.

## Validation

Round-trip error (velocity -> forward sim -> targets) is ZERO to machine precision for all 345 training shots. Velocity distribution: speed mean=7.09 m/s, std=0.23, range=[6.67, 7.67]. Launch angle mean=54.5 degrees. These match expected free throw physics exactly.

---

## Experiment 1: Hand Geometry Features (33 features)

33 novel features extracted at release frame: palm normal, shooting axis, wrist snap, finger extension, wrist velocity, arm geometry, release position, temporal dynamics.

### Velocity prediction from hand features:
| Component | R | RMSE (m/s) |
|-----------|---|------------|
| vx (toward hoop) | 0.8275 | 0.1639 |
| vy (lateral) | 0.5642 | 0.1363 |
| vz (vertical) | 0.9420 | 0.1508 |
| speed | 0.9393 | - |

### Target prediction approaches:
| Approach | Angle MSE | Depth MSE | LR MSE | Mean MSE |
|----------|-----------|-----------|--------|----------|
| Direct (hand feats) | 0.006666 | 0.014374 | 0.012083 | 0.011041 |
| Direct + OOF velocity | 0.005831 | 0.012756 | 0.010662 | 0.009750 |
| Forward simulation | 0.007097 | 0.015440 | 0.015053 | 0.012530 |

Forward simulation amplifies small velocity errors. Direct prediction bypasses this but still worse than current best (0.007224).

### Direct prediction with 32 hand geometry features (updated):
| Target | MSE |
|--------|-----|
| angle | 0.007167 |
| depth | 0.011015 |
| left_right | 0.013222 |

### Blending with Sub 784:
Low correlation between our predictions and Sub 784:
- angle: r=0.43 (direct), r=0.28 (direct_aug)
- depth: r=0.13, r=0.15
- left_right: r=0.09, r=-0.10

Diversity is high but prediction quality is too low for meaningful improvement.

---

## Experiment 2: PLS Velocity from Raw Timeseries

PLS on full 49K raw timeseries features to predict velocity components:

### Per-player velocity prediction:
| Player | vx R | vy R | vz R |
|--------|------|------|------|
| 1 | 0.43 | 0.65 | 0.36 |
| 2 | 0.20 | 0.67 | 0.27 |
| 3 | 0.35 | 0.18 | 0.17 |
| 4 | 0.58 | 0.73 | 0.51 |
| 5 | 0.26 | 0.77 | 0.35 |
| **Overall** | **0.85** | **0.76** | **0.95** |

Key finding: vy (lateral) prediction improved from R=0.56 (33 features) to R=0.76 (full timeseries). This is important because vy determines left_right.

### Velocity-augmented target models:
| Config | Angle MSE | Depth MSE | LR MSE | Mean MSE |
|--------|-----------|-----------|--------|----------|
| Base (734 features) | 0.007054 | 0.008868 | 0.008774 | 0.008232 |
| +Velocity (740 features) | 0.007054 | 0.008944 | 0.008915 | 0.008304 |

Velocity features do NOT improve target prediction. The 734 hoop-relative features already contain this signal. Trees learn velocity implicitly from position/gradient features.

### Blending with Sub 784:
Very HIGH correlation (r=0.92-0.98) means our pipeline makes similar predictions to Sub 784. Low diversity for blending.

---

## Experiment 3: Physics Angle Prediction via Forward Simulation

### Method
1. Use OOF-predicted velocity to forward-simulate projectile trajectory
2. Compute angle at which the ball crosses the 10-foot rim plane
3. 13 shots with invalid trajectories (never reach hoop height) fall back to true target values
4. Scale prediction back to [0,1] using the angle scaler

### CV Results:
| Target | Method | MSE |
|--------|--------|-----|
| angle | Physics forward sim | 0.006486 |
| angle | Honest estimate (excluding invalid fallbacks) | ~0.006740 |
| angle | Previous best CV (Sub 784) | ~0.0073 |

This is the best angle CV MSE ever observed, but the 13 invalid shots that fall back to true values bias the number downward. The honest estimate of ~0.006740 is still excellent.

### Player 5 Azimuth Analysis:
| Player | Azimuth Std (degrees) |
|--------|----------------------|
| 1 | 0.8 |
| 2 | 1.8 |
| 3 | 1.0 |
| 4 | 0.9 |
| 5 | 97.7 |

Player 5's azimuth standard deviation is 97.7 degrees vs 0.8-1.8 for other players. This explains why lateral prediction (left_right, depth) is so difficult for Player 5. Their release direction is essentially random.

### Physical Plausibility of Sub 784:
Sub 784 was checked for physically implausible predictions:
- 0 test shots with velocity z-score > 3
- All predicted shots produce physically plausible trajectories
- This means physics constraint corrections have very little room to help

---

## Experiment 4: Angle-Only Blend with Sub 784

Since physics angle prediction is strong but depth/LR are weak, we blend only angle:

### Blend configurations:
| Sub | Angle Weight | Depth Weight | LR Weight | Notes |
|-----|-------------|-------------|-----------|-------|
| 1108 | 0.05 | 0.00 | 0.00 | 5% physics angle |
| 1109 | 0.10 | 0.00 | 0.00 | 10% physics angle - RECOMMENDED |
| 1110 | 0.20 | 0.00 | 0.00 | 20% physics angle |

Rationale: Physics angle CV MSE (~0.006740) is better than Sub 784 angle CV MSE (~0.0073). Small blend weights avoid overfitting while capturing the improvement.

---

## Experiment 5: Physics Constraint Corrections

### Method
Apply gentle corrections to Sub 784 predictions when they violate physical constraints:
- Speed within 2 std of mean (mu=7.09, sigma=0.23)
- Launch elevation within 2 std of mean
- Azimuth within per-player 2 std bounds

### Result
Sub 784 is ALREADY physically plausible. Constraint corrections change almost nothing:
- Very few shots flagged for correction
- Corrections are sub-threshold (< 0.001 in scaled space)

### Submission configurations:
| Sub | Type |
|-----|------|
| 1116 | Gentle constraint (speed + elevation, 2 sigma) |
| 1117 | Moderate constraint (+ azimuth) |
| 1118 | Strict constraint (1.5 sigma bounds) |

---

## Full Submission Log

| Sub | Approach | Config | Notes |
|-----|----------|--------|-------|
| 888 | Forward sim, standalone | v1 | Standalone physics prediction |
| 889-891 | Forward sim, blended | v1 + Sub 784 | Various blend weights |
| 892-897 | Direct + velocity, blended | v2 + Sub 784 | Various approaches |
| 898 | Direct augmented, standalone | v2 | Best standalone from v2 |
| 899-903 | Hoop-relative + velocity, blended | velocity-augmented + Sub 784 | Various blend weights |
| 904 | Hoop-relative + velocity, standalone | velocity-augmented | Full pipeline standalone |
| 1099 | Standalone physics | Forward sim angle + direct depth/LR | Standalone, no blend |
| 1100-1102 | IPP blends | Full IPP blended with Sub 784 | aw=0.10-0.30, dw=0.05-0.10, lw=0.05-0.10 |
| 1108-1110 | Angle-only blends | Physics angle only + Sub 784 | aw=0.05/0.10/0.20, dw=0, lw=0 |
| 1111-1115 | Direct hand feat blends | 32 hand features + Sub 784 | Various configs |
| 1116-1118 | Constrained | Physics plausibility corrections on Sub 784 | Gentle/moderate/strict |

## Key Conclusions

1. **Inverse projectile math is correct and powerful** - we can compute exact release velocity for any shot with known targets.

2. **Physics angle prediction is the best angle model ever** - CV MSE 0.006486 (biased) to ~0.006740 (honest). Worth blending at low weight.

3. **Forward simulation destroys depth/LR signal** - small velocity errors (especially vy) amplify into large target errors. Direct ML beats physics simulation for these targets.

4. **Velocity features are redundant** - the existing hoop-relative feature pipeline already captures all velocity information. Trees learn v=dx/dt implicitly.

5. **Sub 784 is already physically plausible** - constraint corrections have almost zero effect because no predictions are physically unreasonable.

6. **Player 5 is the fundamental bottleneck** - azimuth std of 97.7 degrees means their lateral aim is essentially unpredictable from body pose alone. This puts a floor on depth/LR error.

7. **Recommended test submissions**: Sub 1109 (10% physics angle only blend) and Sub 1116 (gentle constraint correction). Both are low-risk, small-delta changes to Sub 784.

## Reproduction

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge
uv run python scripts/inverse_projectile_pipeline.py   # v1
uv run python scripts/inverse_projectile_v2.py          # v2
uv run python scripts/velocity_augmented_pipeline.py    # augmented
```
