# Fourier Trajectory Model Results

**Date:** 2026-02-16
**Script:** scripts/fourier_trajectory_model.py

## Objective
Build a novel model using FFT-based features from joint trajectories to produce DIVERSE predictions
compared to the existing Ridge-based pipeline. Frequency-domain features capture rhythm, tempo, and
periodicity of shooting motion - fundamentally different from single-frame position features.

## Method
1. Extract trajectories for 15 key joints (right/left wrist, elbow, shoulder, hip, knee, ankle, mid_hip, nose, neck) x 3 coords
2. Multi-scale FFT at 3 temporal windows:
   - Full motion: frames 60-200 (overall rhythm)
   - Pre-release: frames 100-170 (shot preparation)
   - Release: frames 130-190 (release mechanics)
3. Hann windowing to reduce spectral leakage
4. First 10 Fourier coefficient magnitudes per trajectory (skip DC)
5. Total: 15 joints x 3 coords x 10 coeffs x 3 windows = 1350 raw features
6. Per-player PLS compression (5 components) with honest LOO (PLS refit per fold)
7. Locally-weighted Ridge regression with Gaussian kernel

## Configuration
- Bandwidth: angle=0.80, depth=0.55, LR=0.30 (same as baseline)
- Ridge alpha: 10.0
- PLS components: 5 per player
- Player override: P1 LR bw=0.15

## Results

### Honest LOO (PLS refit per fold)
| Target     | Honest LOO | Per-player breakdown                           |
|------------|------------|-------------------------------------------------|
| angle      | 0.006671   | P1:0.002068 P2:0.005013 P3:0.003368 P4:0.005603 P5:0.016505 |
| depth      | 0.006675   | P1:0.005070 P2:0.007358 P3:0.002149 P4:0.005822 P5:0.012515 |
| left_right | 0.009645   | P1:0.010387 P2:0.009420 P3:0.005015 P4:0.009433 P5:0.013590 |
| **Mean**   | **0.007664** |                                              |

### PLS Leakage Check
| Metric | Value |
|--------|-------|
| Fast (leaky) LOO | 0.001230 |
| Honest LOO | 0.007664 |
| Leakage ratio | 6.23x |

### Diversity vs Sub 2716 (test predictions)
| Target     | Correlation r | Assessment |
|------------|--------------|------------|
| angle      | 0.9643       | High (low diversity) |
| depth      | 0.7389       | GOOD DIVERSITY |
| left_right | 0.8256       | Moderate diversity |
| **Mean**   | **0.8429**   | |

## Kill Check
- Honest LOO 0.007664 < 0.015 threshold: PASS
- Not all correlations > 0.90 (depth r=0.74, LR r=0.83): PASS
- Model proceeds to submission

## Submissions Generated
| Sub # | Description | Notes |
|-------|-------------|-------|
| 2949  | Fourier standalone | Honest LOO 0.007664 |
| 2950  | 2% Fourier + 98% Sub 2716 (all targets) | Most conservative blend |
| 2951  | 2% Fourier angle only + 98% Sub 2716 | Angle has r=0.96 (low diversity) |
| 2952  | 2% Fourier depth only + 98% Sub 2716 | Depth has best diversity (r=0.74) |
| 2953  | 2% Fourier LR only + 98% Sub 2716 | LR has moderate diversity (r=0.83) |
| 2954  | 5% Fourier depth + 95% Sub 2716 | Higher weight on most diverse target |
| 2955  | 10% Fourier depth + 90% Sub 2716 | Aggressive on depth |

## Key Findings
1. **Strong diversity for depth**: r=0.7389 is excellent. The Fourier model captures different
   information about depth than the Ridge baseline.
2. **Moderate diversity for LR**: r=0.8256 provides some blend value.
3. **Low diversity for angle**: r=0.9643 means angle predictions are similar to baseline.
4. **P5 is the weakness**: Consistently worst across all targets (0.012-0.017). Small sample
   effect - FFT features need enough data to reliably estimate frequency content.
5. **Massive PLS leakage**: 6.23x ratio between leaky and honest LOO confirms that PLS
   on small per-player samples is highly prone to leakage.
6. **Honest LOO roughly matches baseline quality**: 0.007664 vs baseline ~0.006830 honest LOO.
   The Fourier model is slightly weaker on its own but provides valuable diversity.

## Priority for LB Testing
1. **Sub 2952** (2% Fourier depth only): Best risk/reward. Depth has lowest correlation.
2. **Sub 2954** (5% Fourier depth): Moderate risk if 2% works.
3. **Sub 2950** (2% all targets): Conservative full blend.

## Reproducibility
- Run: `uv run python scripts/fourier_trajectory_model.py`
- Data: data/train.csv, data/test.csv
- Scalers: data/scaler_angle.pkl, data/scaler_depth.pkl, data/scaler_left_right.pkl
- Anchor: submission/submission_2716.csv
- Runtime: ~25 seconds on M-series Mac
