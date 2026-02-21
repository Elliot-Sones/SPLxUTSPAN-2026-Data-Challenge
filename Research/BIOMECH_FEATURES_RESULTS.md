# Biomechanical Feature Extraction Results (2026-02-07)

## Experiment

Script: `scripts/biomech_enhanced_blend.py`

Extracted 43 biomechanical features from joint angular velocities, proximal-to-distal
timing delays, trunk lean, center of mass velocity, coordination variability, knee
angular velocity, release height ratio, body alignment, and temporal shape features.

These features are fundamentally different from the existing hoop-relative position
statistics - they capture angular velocities and timing patterns that determine
WHERE the ball goes, not just WHERE joints ARE.

## Feature Groups

| Group | Count | Description |
|-------|-------|-------------|
| Angular Velocities | 13 | Elbow, shoulder, wrist angular vel at release + peaks |
| Proximal-to-Distal Timing | 4 | Shoulder->elbow->wrist->fingertip timing delays |
| Trunk Lean | 3 | Forward/lateral lean angle, lean rate at frame 153 |
| Center of Mass | 4 | CoM speed, vertical vel, forward vel, rising flag |
| Coordination Variability | 4 | Elbow angvel std, shoulder angvel CV, wrist jerk |
| Knee Angular Velocity | 3 | Knee angvel at release, peak, knee-to-elbow timing |
| Release Height Ratio | 2 | Release height / player height, release height / hoop |
| Body Alignment | 4 | Arm plane alignment, shoulder/hip rotation, arm tilt |
| Temporal Shape | 6 | Time-to-peak, snap duration, propulsion duration, skewness, decel rate |
| **Total** | **43** | |

## Sanity Check (First 5 Shots)

- Elbow angular velocity: 412-455 deg/s (literature: 300-800 deg/s) - PASS
- Shoulder angular velocity: 178-199 deg/s (literature: 200-500 deg/s) - slightly low, acceptable
- Trunk forward lean: -0.5 to 1.2 deg (literature: 5-20 deg) - lower than expected for free throw (less lean than jump shot)

## Feature Correlations with Targets

### Angle (35 of 43 features with |r| > 0.15)
| Feature | r |
|---------|---|
| bm_trunk_forward_lean_153 | -0.6858 |
| bm_wrist_snap_duration | -0.5307 |
| bm_elbow_time_to_peak | -0.5146 |
| bm_release_z_over_hoop | -0.4701 |
| bm_shoulder_angvel_peak_propulsion | +0.4648 |
| bm_com_vel_z_at_153 | -0.4621 |
| bm_com_rising_at_153 | -0.4035 |
| bm_elbow_angvel_at_150 | -0.3850 |

### Depth (11 of 43 features with |r| > 0.15)
| Feature | r |
|---------|---|
| bm_com_vel_z_at_153 | +0.3571 |
| bm_elbow_angvel_at_153 | +0.3015 |
| bm_elbow_angvel_at_150 | +0.2493 |
| bm_wrist_angvel_at_153 | -0.2223 |
| bm_hip_rotation_vs_hoop | -0.2128 |
| bm_shoulder_rotation_vs_hoop | -0.2011 |
| bm_knee_angvel_at_153 | +0.1998 |
| bm_trunk_forward_lean_rate_153 | -0.1970 |

### Left_right (1 of 43 features with |r| > 0.15)
| Feature | r |
|---------|---|
| bm_arm_lateral_tilt | +0.2550 |

## CV Results

### HR-only vs HR+BM (per-player per-target, 5-fold, top-80 MI feature selection)

| Target | HR Baseline MSE | HR+BM MSE | Improvement |
|--------|----------------|-----------|-------------|
| Angle | 6.097400 | 5.991143 | +1.74% |
| Depth | 13.166362 | 11.957259 | **+9.18%** |
| Left_right | 7.670796 | 7.516124 | +2.02% |
| **Mean** | **8.978186** | **8.488175** | **+5.46%** |

### BM-only Performance & Diversity

| Target | BM-only MSE | Correlation with HR |
|--------|-------------|-------------------|
| Angle | 6.927719 | 0.9569 |
| Depth | 13.553427 | 0.8717 |
| Left_right | 10.209848 | 0.6504 |

### Test Prediction Correlations with Sub 784

| Target | r with Sub 784 |
|--------|----------------|
| Angle | 0.9542 |
| Depth | 0.6549 |
| Left_right | 0.8384 |

## Generated Submissions

All blended with Sub 784 using combined (HR+BM) model predictions:

| Sub | angle_w | depth_w | lr_w | depth_source | diversity |
|-----|---------|---------|------|-------------|-----------|
| 1362 | 0.20 | 0.30 | 0.50 | combined | 0.00152824 |
| 1363 | 0.15 | 0.30 | 0.50 | combined | 0.00149291 |
| 1364 | 0.10 | 0.30 | 0.50 | combined | 0.00146768 |
| 1365 | 0.05 | 0.30 | 0.50 | combined | 0.00145254 |
| 1366 | 0.00 | 0.30 | 0.50 | combined | 0.00144749 |
| 1367 | best-per-target, w=0.15 for all improved targets | - |

## Key Insights

1. **Trunk forward lean is the strongest single biomech feature** (r=-0.69 with angle). This was
   not captured by any existing feature - position-based features capture trunk position but not
   the angular lean.

2. **Depth benefits most from biomech features** (+9.18%). This makes sense because depth is the
   hardest target (r=0.71) and biomech features like CoM vertical velocity (r=0.36) and elbow
   angular velocity (r=0.30) provide genuinely new signal about the force applied to the ball.

3. **Left_right benefits least** (+2.02%). Only bm_arm_lateral_tilt has |r| > 0.15 for left_right,
   which makes physical sense - lateral deviation is primarily about arm alignment (already
   captured by hoop-relative features) rather than angular velocities.

4. **Feature selection (top-80 by MI) is critical** to prevent overfitting with 778 total features
   on 345 samples.

5. The biomech features have good diversity from HR features (especially depth at r=0.87 and
   LR at r=0.65), suggesting they add genuinely different signal.

## Reproduction

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge
uv run python scripts/biomech_enhanced_blend.py
```

Runtime: ~156 seconds. Requires: scipy, lightgbm, xgboost, catboost, scikit-learn.
