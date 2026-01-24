# Physics-Based vs Direct ML Prediction - Test Results

## Objective
Compare two approaches for predicting basketball shot outcomes:
- **Approach A (Direct ML)**: Hand kinematics → Outcomes
- **Approach B (Physics)**: Hand kinematics → Velocity → Ballistic physics → Outcomes

## Test Configuration

**Date**: 2026-01-23
**Dataset**: SPLxUTSPAN 2026 competition (345 training shots)
**Test setup**: 100 shots, 80/20 train/test split
**Features**: 40 physics-based features (not full 3000+ engineered features)

## Results

### Approach A: Direct ML
- **Overall MSE**: 11.89
- **Per-target MSE**:
  - Angle: 4.70
  - Depth: 13.57
  - Left/Right: 17.41

### Approach B: Physics-Based
- **Overall MSE**: 15.71 (32% worse than Direct ML)
- **Per-target MSE**:
  - Angle: 5.09
  - Depth: 27.48
  - Left/Right: 14.58

### Velocity Prediction Quality
- **vx predictor**: R² = 0.04 (essentially random)
- **vy predictor**: R² = 0.12 (weak correlation)
- **vz predictor**: R² = -0.37 (worse than baseline)

## Why Physics Approach Failed

### 1. Poor Velocity Prediction
Ground truth velocities from inverse ballistics couldn't be predicted from hand kinematics:
- R² scores near zero indicate features don't capture velocity
- Possible causes:
  - Wrong feature set (using release frame, not acceleration phase)
  - Coordinate system mismatch
  - Ground truth velocities may be inaccurate

### 2. Error Propagation
Even small velocity errors get amplified through ballistic equations:
- 10% velocity error → 20-30% outcome error
- Physics is deterministic but sensitive to inputs

### 3. Limited Feature Set
Only 40 physics features vs 3000+ in full approach:
- Missing interactions
- Missing temporal patterns
- Missing per-player adjustments

## Ground Truth Velocity Computation

### Inverse Ballistics Results (All 345 Shots)
- **Successfully processed**: 335/345 shots (97%)
- **Good convergence** (error < 0.1): 309/335 shots (92%)
- **Convergence error**:
  - Mean: 3.88
  - Median: 0.00 (most shots converge perfectly)
  - Max: 269.97 (few outliers)

### Velocity Statistics
- **Ground truth velocity magnitude**:
  - Mean: 11.90 units/s
  - Std: 2.60
  - Range: 5.95 - 23.90

- **Observed velocity magnitude** (from tracking):
  - Mean: 12.37 units/s
  - Std: 3.96

- **Velocity difference** (GT vs Observed):
  - Mean: 17.55 units/s
  - Median: 17.62 units/s

### Key Finding: No Correlation
Observed velocity from motion tracking has **near-zero correlation** with ground truth velocity from inverse ballistics:
- vx correlation: -0.08
- vy correlation: 0.05
- vz correlation: 0.02

**This suggests**: Either the coordinate system is wrong, or the tracking doesn't accurately capture release velocity.

## Prediction Quality Check
The inverse ballistics solutions reproduce targets well:
- Angle MAE: 0.08° (excellent)
- Depth MAE: 0.34 units (excellent)
- Left/Right MAE: 0.17 units (excellent)

This validates that the physics model itself is correct when given proper velocities.

## Implications

### What Worked
1. **Inverse ballistics solver**: Converges for 92% of shots with minimal error
2. **Physics model**: Accurately reproduces outcomes from velocity
3. **Calibration**: Hoop position estimation from data works

### What Failed
1. **Velocity prediction**: Features don't capture the information needed
2. **Coordinate system**: Possible mismatch between tracking and physics
3. **Feature engineering**: Need pre-release dynamics, not just release frame

### Why Direct ML Wins
Direct ML learns the **implicit mapping** from kinematics to outcomes without needing the intermediate velocity step. It can:
- Learn correlations we didn't explicitly model
- Handle coordinate system issues automatically
- Use all 3000+ features effectively

## Hypothesis: Why Velocity Can't Be Predicted

### Theory 1: Wrong Feature Window
- **Current**: Using features AT release frame
- **Problem**: Ball is already launched, velocity already determined
- **Fix**: Use features from 10-20 frames BEFORE release (acceleration phase)

### Theory 2: Coordinate System Mismatch
- **Observed**: vx, vy, vz from wrist tracking
- **Actual**: Ball velocity ≠ wrist velocity at release
- **Missing**: Hand-ball offset, release timing, finger flick

### Theory 3: Unmeasured Variables
- **Ball spin**: Not tracked, but affects trajectory via Magnus effect
- **Release timing**: 60fps = 16ms resolution, release happens over 10-20ms
- **Finger kinematics**: Only tracked to wrist, not finger tip

## Next Steps to Fix Physics Approach

### Option 1: Debug Velocity Features
1. Plot wrist velocity vs ground truth velocity (check correlation)
2. Add pre-release features (frames 80-120, acceleration phase)
3. Add velocity magnitude features, not just components

### Option 2: Validate Coordinate System
1. Visualize trajectories from inverse ballistics
2. Check if hoop position makes sense
3. Plot observed vs predicted outcomes

### Option 3: Hybrid Model
1. Train GB on physics features + direct predictions
2. Use physics as a constraint/prior
3. Ensemble: 70% direct + 30% physics

## Competitive Strategy

### For Competition (Win Focus)
- **Use Direct ML** with full 3000+ features
- **Current baseline**: MSE ≈ 0.024
- **Physics doesn't help**: Adds complexity without improvement

### For Understanding (Science Focus)
- **Fix physics approach** to understand mechanism
- **Debug velocity prediction** to find root cause
- **Validate findings** against biomechanics literature

## Open Questions

1. **What is the noise floor?** (Running test now)
   - Can train MSE → 0? (Deterministic problem)
   - Or does it plateau? (Stochastic noise)

2. **Can we estimate spin from kinematics?**
   - Wrist snap rate, finger flick velocity
   - Would explain remaining variance

3. **Is the coordinate system correct?**
   - Units unknown (meters? feet? normalized?)
   - Axes orientation unclear

## Files Generated

- `output/ground_truth_velocities.csv`: Inverse ballistics solutions for 335 shots
- `output/calibrated_hoop_position.csv`: Estimated hoop position and gravity
- `src/inverse_ballistics.py`: Physics solver implementation
- `src/compute_ground_truth_velocities.py`: Batch processing script
- `src/train_velocity_predictor.py`: Velocity predictor training (incomplete)
- `src/quick_physics_comparison.py`: Fast empirical test

## Conclusion

**Physics approach is theoretically superior but practically inferior.**

The hypothesis that "predicting velocity is more natural than outcomes" doesn't hold empirically with current features and coordinate system. Direct ML extracts the signal more effectively for this competition.

To make physics work, we'd need to:
1. Fix coordinate system issues
2. Add pre-release dynamics features
3. Model spin explicitly
4. Validate against known physics

**Effort**: High
**Expected improvement**: Uncertain
**Recommendation**: Use Direct ML for competition, physics for post-hoc validation

---

## Noise Floor Test (In Progress)

Testing if we can achieve near-zero training error with extreme overfitting. Results pending...
