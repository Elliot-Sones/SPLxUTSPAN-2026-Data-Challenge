# Test Results Log - SPL UTSpan Data Challenge 2026

## 2026-02-17: P5 Physics-Leakage-GhostDTW Depth Ablations (NEW)

### Exact Runs

1. `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/p5_physics_leakage_ghost_dtw_ablation.py --scale 1 --seed 20260217 --run-tag p5_phy_leak_ghost_pilot_s1_20260217`
2. `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/p5_physics_leakage_ghost_dtw_ablation.py --scale 3 --seed 20260217 --run-tag p5_phy_leak_ghost_full_s3_20260217`

Data and model details:
- Data: `data/train.csv` (345 rows), `data/test.csv` (113 rows)
- Depth scaling: `data/scaler_depth.pkl`
- Anchor submission: `submission/submission_2716.csv`
- Target modified for submission blends: `scaled_depth` only
- Blend weights evaluated: `0.01`, `0.02`, `0.03`
- Ablations:
  - `leakage_only`
  - `physics_residual`
  - `physics_residual_ghost_dtw`

Pilot (`scale=1`) exact depth OOF MSE:
- leakage_only: `0.012730846764261`
- physics_residual: `0.015116033367132`
- physics_residual_ghost_dtw: `0.012696909499517`

Scaled (`scale=3`) exact depth OOF MSE:
- leakage_only: `0.014339794160017`
- physics_residual: `0.015842555515563`
- physics_residual_ghost_dtw: `0.014729311586163`

Best CV variant in this study:
- `physics_residual_ghost_dtw` at `scale=1` with OOF depth MSE `0.012696909499517`

Generated submission files:
- Pilot (`scale=1`): `submission/submission_3149.csv` to `submission/submission_3157.csv`
- Scaled (`scale=3`): `submission/submission_3158.csv` to `submission/submission_3166.csv`

Top conservative candidates from this run family:
- `submission/submission_3155.csv` (1% depth injection, `physics_residual_ghost_dtw`, scale=1)
- `submission/submission_3152.csv` (1% depth injection, `physics_residual`, scale=1)
- `submission/submission_3149.csv` (1% depth injection, `leakage_only`, scale=1)

Reproducibility artifacts:
- `output/p5_physics_leakage_ghost_dtw_details_p5_phy_leak_ghost_pilot_s1_20260217.md`
- `output/p5_physics_leakage_ghost_dtw_run_p5_phy_leak_ghost_pilot_s1_20260217.json`
- `output/p5_physics_leakage_ghost_dtw_details_p5_phy_leak_ghost_full_s3_20260217.md`
- `output/p5_physics_leakage_ghost_dtw_run_p5_phy_leak_ghost_full_s3_20260217.json`

## Leaderboard Submissions

| Sub # | Date | LB Score | Method | Notes |
|-------|------|----------|--------|-------|
| 219 | - | **0.007682** | Selective Amplification | **BEST** - pctl=91, alpha=1.1, base=Sub133, contrast=Sub151 |
| 183 | - | 0.007698 | Selective Amplification | pctl=90, alpha=1.0 |
| 133 | - | 0.007809 | 4-way blend | 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111 |
| 616 | - | 0.007800 | Target-specific amplification | (93,1.3) all targets |
| 663 | 2026-02-03 | 0.020676 | TabNet only | FAILED - overfit, wrong CV |
| 676 | 2026-02-03 | 0.009254 | TabNet Residual | Residual corrections to Sub219 |
| 714 | 2026-02-03 | 0.007764 | Selective Amp (pctl=93, alpha=1.0) | Worse than pctl=91 |
| 704 | 2026-02-03 | 0.007735 | Selective Amp (pctl=91, alpha=1.0) | Worse than alpha=1.1 |
| 739 | 2026-02-03 | 0.007993 | 10% bone-denoised + 90% Sub219 | CV improvement didn't transfer to LB |
| 752 | 2026-02-03 | 0.010137 | Savitzky-Golay denoised ensemble | **FAILED** - 11% CV improvement, 32% LB degradation |
| 753 | 2026-02-03 | TBD | Top 20 features per target | **18% CV improvement** - test on LB |
| 754 | 2026-02-03 | 0.007686 | 90% Sub219 + 10% Sub753 | Near-best, only 0.000004 worse than Sub219 |
| 755 | 2026-02-03 | TBD | 80% Sub219 + 20% Sub753 | Blend test |
| 756 | 2026-02-03 | TBD | 70% Sub219 + 30% Sub753 | Blend test |

## Pending Submissions (Not Yet Tested on LB)

### Echo State Networks (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 678 | ESN pure | angle_std=0.125, CV MSE 0.053 - poor |
| 679 | 10% ESN + 90% Sub219 | angle_std=0.136 |
| 680 | 20% ESN + 80% Sub219 | angle_std=0.134 |
| 681 | 30% ESN + 70% Sub219 | angle_std=0.133 |

### Temporal Depth Features (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 682 | Pure temporal depth | CV MSE 0.055 for depth |
| 683 | 10% temporal + 90% Sub219 depth | Conservative blend |
| 684 | 20% temporal + 80% Sub219 depth | |
| 685 | 30% temporal + 70% Sub219 depth | |
| 686 | 50% temporal + 50% Sub219 depth | |

### Selective Amplification Grid Search (2026-02-03)
| Sub # | Parameters | angle_std | Notes |
|-------|-----------|-----------|-------|
| 687-691 | pctl=88, alpha=0.8-1.2 | 0.1377-0.1378 | |
| 692-696 | pctl=89, alpha=0.8-1.2 | 0.1376-0.1377 | |
| 697-701 | pctl=90, alpha=0.8-1.2 | 0.1375 | Similar to Sub183 |
| 702-706 | pctl=91, alpha=0.8-1.2 | 0.1374-0.1375 | Similar to Sub219 |
| 707-711 | pctl=92, alpha=0.8-1.2 | 0.1372-0.1373 | |
| **714-716** | **pctl=93, alpha=1.0-1.2** | **0.1371-0.1372** | **Closest to target 0.137** |
| 717-721 | Target-specific params | 0.1372-0.1375 | Various configs |

Recommended to test on LB: Sub 714-716 (pctl=93, closest to target profile)

### Spatial Graph Features (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 722 | Pure spatial features | CV MSE 0.103 - poor, angle_std=0.132 |
| 723 | 10% spatial + 90% Sub219 | Conservative blend |
| 724 | 20% spatial + 80% Sub219 | |
| 725 | 30% spatial + 70% Sub219 | |

### KAN-like Features (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 726 | B-spline KAN-like | angle_std=0.085 - poor profile |
| 727 | 10% B-spline + 90% Sub219 | |
| 728 | 20% B-spline + 80% Sub219 | |
| 729 | 30% B-spline + 70% Sub219 | |

### tsfresh Features (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 730 | Pure tsfresh | angle_std=0.147 - too high |
| 731 | 10% tsfresh + 90% Sub219 | |
| 732 | 20% tsfresh + 80% Sub219 | |
| 733 | 30% tsfresh + 70% Sub219 | |

### MiniRocket Features (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 734 | Pure MiniRocket | angle_std=0.148 |
| 735 | 10% MiniRocket + 90% Sub219 | |
| 736 | 20% MiniRocket + 80% Sub219 | |
| 737 | 30% MiniRocket + 70% Sub219 | |

### Bone Length Optimization (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 738 | Bone-denoised features | angle_std=0.129, CV improved 9% |
| 739 | 10% bone-denoised + 90% Sub219 | |
| 740 | 20% bone-denoised + 80% Sub219 | |
| 741 | 30% bone-denoised + 70% Sub219 | |

### Target-Specific Ensemble (2026-02-03)
| Sub # | Method | Notes |
|-------|--------|-------|
| 742 | Pure ensemble (KAN angle + temporal depth + Sub219 lr) | angle_std=0.128 |
| 743-745 | 30% KAN angle + varying depth | |
| 746-748 | 50% KAN angle + varying depth | |
| 749-751 | 70% KAN angle + varying depth | |

## Cross-Validation Results (Not Submitted)

### Echo State Networks (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| ESN (n_res=300, sr=0.95) | GroupKFold | 0.047 | 0.052 | 0.059 | Total: 0.053 - much worse than XGBoost |

### KAN-like Methods (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| B-spline (7 knots, deg 3) | GroupKFold | 0.032 | 0.025 | 0.027 | Total: 0.028 |
| Polynomial (deg 2) | GroupKFold | 0.015 | 0.030 | 0.031 | Total: 0.025 - best KAN-like |
| RBF (200 components) | GroupKFold | 0.031 | 0.024 | 0.027 | Total: 0.027 |
| Actual KAN [20,8,1] | GroupKFold | **0.010** | 0.036 | 0.041 | Total: 0.029 - excellent angle! |

### tsfresh Automated Features (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Minimal (150 features) | GroupKFold | 0.046 | 0.052 | 0.064 | Total: 0.054 |
| Efficient (11,655 features) | GroupKFold | 0.045 | 0.051 | 0.097 | Total: 0.064 - overfitting lr |
| **Selected (20 features)** | GroupKFold | **0.032** | **0.042** | **0.049** | **Total: 0.041 - best tsfresh** |

#### tsfresh Within-Player Correlations
| Target | Best tsfresh Feature | Within-Player r | Notes |
|--------|---------------------|-----------------|-------|
| angle | elbow_y linear trend stderr | r=+0.25 | Still weak |
| depth | mid_hip_z location_of_maximum | **r=+0.56** | Confirms temporal signal! |
| left_right | shoulder_y Dickey-Fuller | r=+0.30 | Moderate |

**Key tsfresh Finding**: The best depth predictor is `mid_hip_z__last_location_of_maximum` (r=0.56) - essentially detecting timing of the shot. This confirms temporal features are key for depth.

### MiniRocket (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Default (alpha=10) | GroupKFold | 0.020 | 0.072 | 0.069 | Total: 0.054 |
| Alpha=1000 | GroupKFold | 0.022 | 0.060 | 0.052 | Total: 0.045 |
| Core window (80-160) | GroupKFold | 0.025 | 0.050 | 0.061 | Total: 0.045 - best depth |

### Bone Length Optimization (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Original features | GroupKFold | 0.038 | 0.054 | 0.035 | Total: 0.042 |
| **Denoised features** | GroupKFold | **0.035** | **0.047** | **0.033** | **Total: 0.038 - 9% improvement!** |
| Combined (+ bone lengths) | GroupKFold | 0.027 | 0.073 | 0.043 | Total: 0.048 |

**Key Bone Finding**: Anatomical denoising (enforcing constant bone lengths) improves CV by 9%, especially for depth (13% improvement). This physics-informed constraint helps remove measurement noise.

### Target-Specific Ensemble (2026-02-03)
| Component | CV Type | MSE | Notes |
|-----------|---------|-----|-------|
| KAN for angle | GroupKFold | 0.024 | High variance (0.004-0.039 per fold) |
| Temporal for depth | GroupKFold | 0.043 | Uses release_frame, set_point_frame |
| Sub219 for left_right | - | - | Best available baseline |

**Insight**: KAN shows excellent performance on some player groups (CV 0.004) but poor on others (0.039), suggesting player-specific patterns that KAN captures well for some but not all.

### Spatial Graph Features (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Ridge on 191 spatial features | GroupKFold | 0.022 | 0.224 | 0.065 | Total: 0.103 - poor |

#### Spatial Features Within-Player Correlations
| Target | Best Spatial Feature | Within-Player r | Notes |
|--------|---------------------|-----------------|-------|
| angle | ankle_to_neck_x | r=-0.15 | Weak - spatial doesn't help |
| depth | release_frame (temporal!) | r=+0.51 | Best predictor is temporal, not spatial |
| left_right | shoulder_forward | r=+0.17 | Weak signal |

**Conclusion**: Spatial graph features (bone lengths, body angles, relative positions) do NOT show strong within-player correlations. Body configuration is not predictive.

### Advanced Biomechanics Features (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Ridge on 35 biomech features | GroupKFold | 0.016 | 0.151 | 0.032 | Total: 0.066 |

### Temporal Depth Features (2026-02-03)
| Experiment | CV Type | depth MSE | Notes |
|------------|---------|-----------|-------|
| Ridge on temporal features | GroupKFold | 0.055 +/- 0.057 | High variance across folds |

#### Within-Player Correlations (Key Finding)
| Target | Best Overall Corr | Best Within-Player Corr | Notes |
|--------|------------------|------------------------|-------|
| angle | release_vz (r=0.78) | release_angle (r=0.09) | Physics features do NOT generalize within-player |
| depth | knee_angle_at_release (r=0.28) | set_point_frame (r=0.56) | TEMPORAL features help within-player |
| left_right | release_vx (r=-0.11) | release_vy (r=0.18) | Weak signal overall |

### Physics Features
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| vz_at_peak correlation | Overall | r=0.78 | r=0.18 | r=0.18 | Strong between-player |
| vz_at_peak correlation | Within-player | r=0.08 | - | - | Weak - doesn't generalize |
| Physics model | LOPO | 0.024 | 0.069 | 0.030 | Physics features alone |

### Rigorous Velocity Extraction (2026-02-03)
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Total | Notes |
|------------|---------|-----------|-----------|--------|-------|-------|
| 39 kinematic features | GroupKFold | 0.056 | 0.028 | 0.019 | 0.035 | 4.5x worse than best |
| 39 kinematic features | KFold | 0.010 | 0.012 | 0.015 | 0.012 | 62% worse than best (0.0077) |

#### Rigorous Velocity Key Features
| Target | Top Features (by correlation) | Correlation |
|--------|------------------------------|-------------|
| angle | elbow_angle_deg | r=+0.60 |
| angle | release_pos_x_ft | r=+0.49 |
| angle | fingertip_acc_mag_fts2 | r=+0.40 |
| depth | launch_angle_deg | r=+0.23 |
| depth | release_vel_z_fts | r=+0.20 |
| left_right | max_fingertip_speed_fts | r=-0.21 |

**Key Finding**: Mocap fingertip velocity (3 m/s) is much lower than required ball velocity (7-8 m/s). This is because:
1. 60Hz sampling misses rapid motion peaks
2. Ball gains additional velocity from arm extension and finger flick not captured by fingertip tracking
3. Ball is not tracked directly, only inferred from hand position

Release detection using peak velocity toward hoop (frame ~105-112) is more accurate than peak height (frame ~150+).

Physics-only features perform poorly on GroupKFold (leave-one-player-out) because they don't generalize across players - each player has unique kinematics.

### TabNet Experiments
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| TabNet original | KFold | 0.011 | 0.024 | 0.030 | Misleading - within-player CV |
| TabNet + GroupKFold | GroupKFold | 0.024 | 0.023 | 0.030 | Proper CV |
| TabNet + feature selection | GroupKFold | 0.014 | 0.027 | 0.046 | Top 50 features |
| TabNet residual | GroupKFold | 0.009 | 0.023 | 0.026 | Best CV, but LB=0.0093 |
| TabNet minimal | GroupKFold | 0.066 | 0.042 | 0.041 | Too restrictive |

### Ensemble/Stacking
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| Single LightGBM | GroupKFold | 0.022 | 0.070 | 0.031 | Baseline |
| Stacking (LGB+XGB+Cat+Ridge) | GroupKFold | 0.040 | 0.024 | 0.027 | Hurts angle, helps depth/lr |

### Augmentation
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Notes |
|------------|---------|-----------|-----------|--------|-------|
| No augmentation | GroupKFold | 0.021 | 0.069 | 0.030 | Baseline |
| With augmentation | GroupKFold | 0.021 | 0.058 | 0.031 | Helps depth +17% |

## Key Findings

### What Works
1. **Selective Amplification** - Best LB scores (0.0077-0.0078)
2. **Heavily regularized gradient boosting** - LightGBM, XGBoost
3. **GroupKFold CV** - Prevents subject leakage
4. **Profile matching** - angle_std ~0.137, depth_mean = 0.5055

### What Doesn't Work
1. **TabNet standalone** - Overfits despite regularization (LB=0.0207)
2. **Physics features for angle** - Strong overall correlation (r=0.78) but weak within-player (r=0.09)
3. **Echo State Networks** - CV MSE 0.053, much worse than gradient boosting
4. **Neural networks** - Insufficient data (345 samples)
5. **Stacking for angle** - Hurts performance
6. **Spatial graph features** - Bone lengths, body angles, relative positions have weak within-player signal (r<0.17)

### Key Insight: Within-Player vs Between-Player Signal
- **Physics features** (velocity, height, etc.) capture between-player differences but NOT within-player variation
- **Temporal features** (set_point_frame, release_frame) have better within-player signal for depth (r=0.56)
- This explains why physics-based models fail to improve LB despite strong overall correlations

### All Major Approaches Tested

All proposed approaches have been comprehensively tested:

| Approach | Result | Notes |
|----------|--------|-------|
| Echo State Networks | Poor (0.053 CV) | Much worse than gradient boosting |
| Spatial Graph Features | Poor (0.103 CV) | Weak within-player signal |
| KANs | Mixed (0.029 CV) | Excellent angle (0.010) but poor depth/lr |
| tsfresh | Moderate (0.041 CV) | Confirms temporal features key for depth |
| MiniRocket | Moderate (0.045 CV) | Best depth with core window |
| **Bone Length Denoising** | **Good (0.038 CV)** | **9% improvement over original!** |

**Key Findings**:
- KANs: Excellent angle CV (0.010) but poor depth/left_right
- tsfresh: Confirms temporal features key for depth (r=0.56 within-player)
- Bone denoising: 9% CV improvement, especially for depth (13%)
- ST-GCN spatial: Weak within-player signal, not promising

## Submission Files Reference

- Best: submission_219.csv (LB: 0.007682)
- Baseline: submission_133.csv (LB: 0.007809)
- TabNet residual: submission_676.csv (LB: 0.009254)
- ESN blends: submission_679-681.csv
- Temporal depth: submission_682-686.csv
- Grid search: submission_687-721.csv (most promising: 714-716)
- Spatial features: submission_722-725.csv (poor CV, not recommended)
- KAN-like: submission_726-729.csv (interesting angle CV but poor profile)
- tsfresh: submission_730-733.csv (confirms temporal signal for depth)
- MiniRocket: submission_734-737.csv (moderate CV)
- **Bone denoised: submission_738-741.csv (9% CV improvement - promising!)**
- Target-specific ensemble: submission_742-751.csv (KAN angle + temporal depth + Sub219 lr)

## Untested Approaches Experiments (2026-02-03)

Testing approaches recommended in research document that were not previously tested.

### Signal Processing Features
| Experiment | CV Type | Total MSE | vs Baseline | Notes |
|------------|---------|-----------|-------------|-------|
| Raw mean/std + LGB | GroupKFold | 33.96 | baseline | Simple features |
| **Savitzky-Golay + LGB** | GroupKFold | **30.22** | **-11%** | **Winner - preserves peaks** |
| Wavelet (db4, level 3) + LGB | GroupKFold | 32.60 | -4% | Moderate improvement |

**Key Finding**: Savitzky-Golay denoising (window=11, polyorder=3) improves CV by 11%. This preprocessing preserves kinematic peaks better than raw data while reducing noise.

### Neural Network Architectures
| Experiment | CV Type | angle MSE | depth MSE | lr MSE | Total MSE | Notes |
|------------|---------|-----------|-----------|--------|-----------|-------|
| TCN (channels=[32,32], dropout=0.5) | GroupKFold | 572.70 | 41.74 | 14.61 | 209.68 | **Severe overfit** |
| Transformer (d_model=32, layers=2, dropout=0.5) | GroupKFold | 51.19 | 31.14 | 14.52 | 32.28 | Slight improvement |
| MC Dropout MLP (hidden=[128,64], dropout=0.3) | GroupKFold | 387.23 | 30.90 | 15.02 | 144.39 | High overfit |

**Key Finding**: Neural networks severely overfit with 345 samples despite heavy regularization. TCN is 6x worse than baseline. Transformer is comparable but not better.

### MC Dropout Uncertainty Analysis
| Target | Uncertainty-Error Correlation |
|--------|------------------------------|
| angle | r = 0.83 (excellent calibration) |
| depth | r = 0.08 (poor) |
| left_right | r = 0.08 (poor) |

**Key Finding**: MC Dropout provides well-calibrated uncertainty for angle (r=0.83), meaning high uncertainty predictions are likely to have high error. This could be used for selective prediction (trust low-uncertainty predictions more).

### Conclusions from Untested Approaches

1. **Savitzky-Golay denoising is promising** - 11% CV improvement, worth testing on LB
2. **Neural architectures fail** - All tested architectures (TCN, Transformer, MC Dropout MLP) overfit severely
3. **Wavelets provide marginal benefit** - 4% improvement, not as good as Savitzky-Golay
4. **Uncertainty quantification works for angle** - r=0.83 correlation suggests reliable confidence estimation

**Recommendation**: Apply Savitzky-Golay preprocessing to the existing best features and test on LB

### Kaggle Winner Techniques (2026-02-03)

Testing techniques used by winners in similar small-data competitions.

| Technique | Result | Notes |
|-----------|--------|-------|
| Adversarial Validation | AUC = 0.49 | Train/test distributions ARE similar - not the cause of CV-LB gap |
| Pseudo-labeling | Inconclusive | Evaluation needs fixing |
| Multi-seed ensembling | 0% change | No benefit from seed averaging |
| **Feature Selection (top 20)** | **-18% MSE** | **WINNER - fewer features reduces overfitting** |

**Key Finding**: Aggressive feature selection (top 20 features per target using mutual information) reduces CV MSE by 18%. This is the most promising technique found.

**Submissions to test**:
- Sub 753: Pure top-20 features
- Sub 754-756: Blends with Sub 219
