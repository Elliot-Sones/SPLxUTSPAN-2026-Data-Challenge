# Test Results

## Quick Augmentation Test: 5x Data with Moderate Augmentation

**Date**: 2026-01-22

### Objective
Test whether moderate augmentation with 5x data helps LightGBM performance.

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | LightGBM (n_estimators=100, max_depth=6, learning_rate=0.1, num_leaves=31) |
| Features | Baseline: mean + last value per column (414 features) |
| Augmentation rotation | +/- 1.0 degrees around z-axis |
| Augmentation noise | std = 0.001 * feature_range |
| Data multiplier | 5x (4 augmented per original sample) |
| Original samples | 345 |
| Augmented samples | 1,725 |
| CV strategy | GroupKFold by participant_id, 5 folds |
| Random seed | 42 |

### Script
```
src/data_augmentation_test/run_quick_test.py
```

Command:
```bash
uv run python src/data_augmentation_test/run_quick_test.py
```

### Results

| Dataset | Samples | MSE |
|---------|---------|-----|
| Original | 345 | 0.023665 |
| Augmented (5x) | 1,725 | 0.025847 |

**Difference**: +0.002182 (+9.2%)

**Verdict**: HURTS

### Conclusion

More augmented data (5x) with moderate augmentation (rotation +/- 1 deg, noise 0.001) **hurts** LightGBM performance. The MSE increased from 0.023665 to 0.025847, a 9.2% degradation.

This confirms the prior result from the conservative augmentation test: data augmentation does not help gradient boosting models on this dataset.

### Interpretation

Possible reasons why augmentation hurts:
1. GBDT models already generalize well with 345 samples
2. Augmented samples may introduce noise that interferes with the decision boundaries
3. The feature extraction (mean + last) may amplify augmentation artifacts
4. The validation targets remain unchanged, so augmented data doesn't add new information about target variance

---

## Previous Test Reference

### Augmentation Test with All 7 Models (Conservative Settings)

**Configuration**:
- Augmentation: rotation +/- 0.1 deg, noise std = 0.0001 * range
- Data: 2x (1 augmented per original)
- Models: LightGBM, XGBoost, CatBoost, RandomForest, Ridge, k-NN, MLP

**Result**: See `output/augmentation_test_results.csv`

LightGBM baseline MSE from that test: ~0.024

---

## Regression Coefficient Analysis (Partial Derivatives)

**Date**: 2026-01-22

### Objective
Compute regression coefficients (beta = dY/dX) to answer: "If feature X changes by 1 unit, how much does target Y change?"

### Configuration

| Parameter | Value |
|-----------|-------|
| Data | 345 training shots, 5 participants |
| Features | 207 keypoint columns x 2 aggregations (mean, std) = 414 features |
| Targets | angle, depth, left_right |
| Method | scipy.stats.linregress for simple regression |
| Multivariate | Ridge regression (alpha=100.0) with top 5 significant features (p<0.01) |

### Script
```
src/regression_analysis.py
```

Command:
```bash
uv run python src/regression_analysis.py
```

### Output Files
- `output/regression_coefficients.csv` - 1,242 feature-target pairs with beta, SE, p-value, R2
- `output/regression_per_player.csv` - 6,210 player-specific regressions
- `output/multivariate_regression.csv` - Multi-feature models for each target
- `output/regression_summaries/` - Top features by target and player
- `output/regression_interpretation.txt` - Plain English interpretation

### Verification
All three verification checks passed:
1. Standardized beta = Pearson r (max diff: 0.000000)
2. R-squared = r^2 (max diff: 0.000000)
3. Cross-check with existing correlations (max diff: 0.000000)

### Key Results

#### Simple Regression: Top Predictors by Target

**ANGLE** (units: degrees)

| Feature | Type | Beta | 95% CI | R2 | Interpretation |
|---------|------|------|--------|-----|----------------|
| left_big_toe_z | mean | -107.73 | [-131.85, -83.61] | 0.183 | 1 unit higher -> 108 deg lower angle |
| left_small_toe_z | mean | -96.96 | [-122.06, -71.86] | 0.143 | 1 unit higher -> 97 deg lower angle |
| right_small_toe_z | mean | -85.18 | [-105.77, -64.59] | 0.161 | 1 unit higher -> 85 deg lower angle |
| left_ankle_z | mean | -43.14 | [-50.97, -35.32] | 0.254 | 1 unit higher -> 43 deg lower angle |

**DEPTH** (units: depth units)

| Feature | Type | Beta | 95% CI | R2 | Interpretation |
|---------|------|------|--------|-----|----------------|
| right_heel_z | mean | 23.09 | [10.75, 35.43] | 0.038 | 1 unit higher -> 23 units deeper |
| right_ankle_z | mean | 18.40 | [8.32, 28.48] | 0.036 | 1 unit higher -> 18 units deeper |
| left_heel_z | mean | 15.84 | [3.55, 28.12] | 0.018 | 1 unit higher -> 16 units deeper |

**LEFT_RIGHT** (weak predictability)

| Feature | Type | Beta | 95% CI | R2 | Interpretation |
|---------|------|------|--------|-----|----------------|
| right_small_toe_z | mean | 19.17 | [1.75, 36.59] | 0.013 | Weak effect |
| right_elbow_z | std | 3.27 | [0.38, 6.15] | 0.014 | Weak effect |

#### Multivariate Regression Performance

| Target | Train R2 | CV R2 (5-fold) | n_features |
|--------|----------|----------------|------------|
| angle | 0.205 | -2.11 (+/- 0.48) | 5 |
| depth | 0.044 | -0.13 (+/- 0.16) | 5 |
| left_right | 0.021 | -0.00 (+/- 0.01) | 5 |

**Note**: Negative CV R2 indicates overfitting due to small sample size (345) and correlated features. Simple regression betas are more reliable for interpretation.

#### Per-Player Variation

Players show different biomechanical patterns. Example for ANGLE prediction:

| Player | Top Feature | Beta | p-value | R2 |
|--------|-------------|------|---------|-----|
| 1 | right_big_toe_z (std) | +30.6 | 0.059 | 0.051 |
| 2 | left_shoulder_z (std) | -63.0 | 0.017 | 0.086 |
| 3 | right_small_toe_z (mean) | -124.0 | 0.0002 | 0.190 |
| 4 | left_knee_z (mean) | +124.5 | 0.003 | 0.128 |
| 5 | left_knee_z (mean) | -57.5 | 0.090 | 0.039 |

### Conclusions

1. **ANGLE is most predictable**: Best single-feature R2 = 0.254 (left_ankle_z mean)
2. **Toe/ankle Z-position dominates**: Higher feet position -> lower launch angle
3. **DEPTH and LEFT_RIGHT are weakly predictable**: Max R2 ~0.04 and ~0.02 respectively
4. **Player-specific models differ**: Player 3 and 4 show opposite signs for knee_z -> angle
5. **Multivariate models overfit**: With 345 samples, combining features hurts generalization

---

## Frame-by-Frame Regression Analysis

**Date**: 2026-01-22

### Objective
Compute regression coefficients per frame to answer:
- "Which frames during the shot are most predictive of the outcome?"
- "When is the release frame?" (should show peak R2)
- "Does post-release pose matter?" (should show low R2)

### Configuration

| Parameter | Value |
|-----------|-------|
| Data | 345 training shots, 5 participants |
| Features | 207 keypoint columns per frame |
| Targets | angle, depth, left_right |
| Frames | 240 (at 60 fps = 4 seconds) |
| Method | scipy.stats.linregress for each frame x feature x target |

### Script
```
src/frame_regression_analysis.py
```

Command:
```bash
uv run python src/frame_regression_analysis.py
```

### Four Analyses Computed

| Analysis | Dimensions | Regressions | Time |
|----------|------------|-------------|------|
| 1. Pooled per-frame | 240 frames x 207 features x 3 targets | 149,040 | 19.7s |
| 2. Per-player per-frame | 240 frames x 5 players x 207 features x 3 targets | 745,200 | 94.4s |
| 3. Binned frames (10-frame windows) | 24 bins x 5 players x 207 features x 3 targets | 74,520 | 9.5s |
| 4. Key frames only | 5 frames x 5 players x 207 features x 3 targets | 15,525 | 2.0s |
| **Total** | | **984,285** | **141.2s** |

### Output Files
- `output/frame_regression_pooled.csv` - 149,040 rows
- `output/frame_regression_per_player.csv` - 745,200 rows
- `output/frame_regression_binned.csv` - 74,520 rows
- `output/frame_regression_key_frames.csv` - 15,525 rows
- `output/frame_r2_summary.csv` - 720 rows (R2 summary per frame per target)
- `output/frame_phase_summary.csv` - 12 rows (phase summary)

### Key Results

#### Estimated Release Frames (Peak Mean R2)

| Target | Frame | Time (s) | Peak Mean R2 |
|--------|-------|----------|--------------|
| angle | 153 | 2.55 | 0.1452 |
| depth | 102 | 1.70 | 0.0220 |
| left_right | 237 | 3.95 | 0.0091 |

**Note**: The "release frame" varies by target, suggesting different biomechanical phases matter for different outcomes.

#### R2 Trend by Frame (Mean R2 Across All Features)

**ANGLE**:
| Frame | 0 | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 239 |
|-------|-----|------|------|------|------|------|------|------|------|
| Mean R2 | 0.0366 | 0.0704 | 0.0713 | 0.1432 | 0.0460 | 0.1366 | 0.0957 | 0.0306 | 0.0406 |

**DEPTH**:
| Frame | 0 | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 239 |
|-------|-----|------|------|------|------|------|------|------|------|
| Mean R2 | 0.0117 | 0.0164 | 0.0089 | 0.0138 | 0.0091 | 0.0058 | 0.0211 | 0.0061 | 0.0072 |

**LEFT_RIGHT**:
| Frame | 0 | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 239 |
|-------|-----|------|------|------|------|------|------|------|------|
| Mean R2 | 0.0036 | 0.0019 | 0.0018 | 0.0036 | 0.0045 | 0.0030 | 0.0068 | 0.0031 | 0.0075 |

#### Phase Summary (Best Feature per Phase)

| Phase | Frames | Target | Best Feature | R2 |
|-------|--------|--------|--------------|-----|
| Phase1_Setup | 0-59 | angle | left_eye_x | 0.421 |
| Phase1_Setup | 0-59 | depth | left_eye_z | 0.051 |
| Phase1_Setup | 0-59 | left_right | right_hip_x | 0.016 |
| Phase2_Windup | 60-119 | angle | right_shoulder_z | 0.306 |
| Phase2_Windup | 60-119 | depth | left_big_toe_x | 0.053 |
| Phase2_Windup | 60-119 | left_right | right_heel_x | 0.011 |
| Phase3_Release | 120-179 | angle | right_elbow_x | 0.308 |
| Phase3_Release | 120-179 | depth | left_small_toe_x | 0.038 |
| Phase3_Release | 120-179 | left_right | right_wrist_y | 0.036 |
| Phase4_Follow | 180-239 | angle | right_ear_z | 0.236 |
| Phase4_Follow | 180-239 | depth | left_ear_z | 0.032 |
| Phase4_Follow | 180-239 | left_right | right_big_toe_z | 0.019 |

#### Top Features at Estimated Release Frame

**ANGLE (Frame 153)**:
| Feature | R2 | p-value |
|---------|-----|---------|
| left_ankle_z | 0.4497 | <0.0001 |
| right_knee_z | 0.4387 | <0.0001 |
| left_knee_z | 0.4321 | <0.0001 |
| left_heel_z | 0.3765 | <0.0001 |
| right_ankle_z | 0.3652 | <0.0001 |

**DEPTH (Frame 102)**:
| Feature | R2 | p-value |
|---------|-----|---------|
| left_second_finger_pip_x | 0.0840 | <0.0001 |
| left_first_finger_cmc_x | 0.0804 | <0.0001 |
| left_first_finger_mcp_x | 0.0789 | <0.0001 |
| left_thumb_x | 0.0789 | <0.0001 |
| left_second_finger_mcp_x | 0.0779 | <0.0001 |

**LEFT_RIGHT (Frame 237)**:
| Feature | R2 | p-value |
|---------|-----|---------|
| right_third_finger_distal_z | 0.0254 | 0.0033 |
| right_third_finger_dip_z | 0.0251 | 0.0035 |
| right_second_finger_dip_z | 0.0251 | 0.0036 |
| right_third_finger_pip_z | 0.0247 | 0.0038 |
| right_first_finger_mcp_z | 0.0245 | 0.0040 |

### Findings vs Hypothesis

**Original Hypothesis**: R2 should increase toward release (frame ~120), then decrease after the ball leaves the hand.

**Actual Findings**:

1. **ANGLE shows a bimodal pattern**: Two peaks around frames 90 and 150, not a single release peak. The best predictability (R2=0.45 for left_ankle_z) occurs at frame 153, well after the expected release point.

2. **Lower body dominates ANGLE prediction**: Ankle and knee z-coordinates at frame 153 explain up to 45% of variance in shot angle. This suggests the shooter's stance/posture during follow-through correlates with launch angle.

3. **DEPTH is weakly predictable throughout**: No clear temporal pattern. Left hand finger positions (non-shooting hand) at frame 102 are most predictive (R2~0.08).

4. **LEFT_RIGHT is essentially unpredictable**: Peak R2 = 0.025 at frame 237 (end of shot). Right hand finger z-coordinates have marginal predictive power.

5. **Post-release frames still matter for ANGLE**: R2 does NOT drop sharply after release. This suggests either:
   - The release frame is later than assumed (>frame 150)
   - Follow-through posture is correlated with release mechanics

6. **Phase analysis shows shifting predictors**:
   - Setup: Eye position predicts angle (R2=0.42)
   - Windup: Shoulder z predicts angle (R2=0.31)
   - Release: Elbow x predicts angle (R2=0.31)
   - Follow: Ear z predicts angle (R2=0.24)

### Conclusions

1. **ANGLE is highly predictable at specific frames**: R2 up to 0.45 at frame 153, using leg position (ankle/knee z)
2. **The "release frame" varies by target**: 153 for angle, 102 for depth, 237 for left_right
3. **Post-release frames are surprisingly predictive**: Does not follow expected "peak at release, drop after" pattern
4. **DEPTH and LEFT_RIGHT remain weakly predictable** across all frames (max R2 < 0.1)
5. **Feature importance shifts across phases**: Eye -> Shoulder -> Elbow -> Ear for angle prediction
# Test Results

## 2026-02-09 - Ball velocity back-solver and v0 predictor (random_forest)

Command:
`python src/train_ball_velocity.py --max-shots 50 --model random_forest --folds 3`

Data:
- First 50 shots from `data/train.csv` via `iterate_shots(train=True, chunk_size=25)`
- Release detection: peak right-wrist speed after frame 80
- Smoothing: 5-frame moving average
- v0 back-solver: rim-plane angle matching via 1-D root search on flight time

Model:
- `MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1))`

CV:
- KFold (fallback because only one participant in sample)
- folds = 3, shuffle = True, random_state = 42

Results:
- raw_mse = 6677956.930257507
- scaled_mse = 6501.255699876078

Command:
`python src/train_ball_velocity.py --max-shots 344 --model random_forest --folds 5`

Data:
- First 344 shots from `data/train.csv` via `iterate_shots(train=True, chunk_size=25)`
- Same release detection, smoothing, and v0 solver as above

CV:
- GroupKFold by participant, folds = 5

Results:
- raw_mse = 8440048.234724633
- scaled_mse = 8226.463644460944

Notes:
- These runs demonstrate that a naive wrist-based r0/v0 estimate is not sufficient; calibration and better ball center estimation are required.

## 2026-02-09 - Angle sensitivity to release velocity (back-solved v0)

Command:
`PYTHONPATH=src python - <<'PY' ...` (script using `build_dataset` and `targets_from_state`)

Data:
- First 200 shots from `data/train.csv` via `build_dataset(max_shots=200, smooth_window=5, t_min=0.1, t_max=2.0, t_steps=200)`
- v0 back-solved from targets using the ballistic rim-plane angle constraint
- r0 = right-wrist position at release (no wrist-to-ball offset calibration)

Method:
- Finite-difference sensitivity with epsilon = 0.1 ft/s:
  - |d(angle)/d(v)| = sqrt( (dθ/dvx)^2 + (dθ/dvy)^2 + (dθ/dvz)^2 )

Results:
- Shots used = 200
- Sensitivity |d(angle)/d(v)| (deg per ft/s):
  - mean = 0.443127
  - median = 0.431899
  - p10 = 0.407214
  - p90 = 0.500400
- Required |v| error for 0.01 deg angle error (ft/s):
  - mean = 0.022719
  - median = 0.023154
  - p10 = 0.019984
  - p90 = 0.024557

Notes:
- Sensitivity is approximate because r0 is estimated from the wrist and no ball-offset calibration is applied.

## 2026-02-09 - Release detector comparison (wrist_speed vs wrist_snap)

Command:
`python src/evaluate_release_detectors.py --max-shots 300 --folds 5`

Data:
- First 300 shots from `data/train.csv` via `iterate_shots(train=True, chunk_size=25)`
- Features: release-phase positions/velocities for wrist, elbow, shoulder, hip; joint angles; trunk lean
- Targets: angle, depth (raw units)

Release detectors:
- wrist_speed: peak right-wrist speed after frame 80
- wrist_snap: peak wrist angular velocity (forearm vs finger) with elbow angle > 140 deg and wrist_z > shoulder_z

Model:
- StandardScaler + MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))

CV:
- GroupKFold by participant, folds = 5

Results:
- wrist_speed:
  - mse_angle = 49.47856851724987
  - mse_depth = 48.86490681579858
  - mse_scaled_angle_depth = 0.04133869079070532
  - n_shots = 300
- wrist_snap:
  - mse_angle = 74.10244101498348
  - mse_depth = 355.1310952011619
  - mse_scaled_angle_depth = 0.1418287640562726
  - n_shots = 300

Notes:
- Under this controlled comparison (same features/model/CV), wrist_speed outperformed wrist_snap.

## 2026-02-09 - Release detector with fingertip-based ball center

Command:
`python src/evaluate_release_detectors.py --max-shots 300 --folds 5`

Data/Features:
- Same as prior release-detector comparison
- Added fingertip-based ball center features:
  - Fingertip centroid + palm normal (index/pinky/wrist) + ball radius (4.7 in)
  - Ball center position and velocity at release

Release detector:
- arm_straight_ball: release frame = max elbow extension angle within window, with fingertip ball center features

Results:
- arm_straight_ball:
  - mse_angle = 129.2212777336614
  - mse_depth = 187.5780661883525
  - mse_scaled_angle_depth = 0.12495798484873266
  - n_shots = 300

Notes:
- The elbow-extension heuristic with fingertip ball center features performed worse than wrist_speed in this baseline setup.

## 2026-02-09 - Arm straight + wrist snap (noise-gated)

Command:
`python src/evaluate_release_detectors.py --max-shots 300 --folds 5`

Detector:
- arm_straight_snap: peak wrist angular velocity with gates:
  - elbow angle > 140 deg
  - wrist_z > shoulder_z
  - wrist speed >= 70th percentile in search window
  - wrist angle smoothed before differentiation

Results:
- arm_straight_snap:
  - mse_angle = 54.945906039750966
  - mse_depth = 170.8570217525202
  - mse_scaled_angle_depth = 0.07895436439638095
  - n_shots = 300

Notes:
- arm_straight_snap improves over wrist_snap but still underperforms wrist_speed on angle and depth in this setup.

---

## 2026-01-28 - Per-Target Independent Hyperparameter Tuning Experiment

**Date**: 2026-01-28

### Objective

Test whether per-target hyperparameter tuning improves over shared hyperparameters. The previous S1/S2/S3/S4 strategy comparison showed S1=S2 and S3=S4 because all used identical hyperparameters - this experiment properly tests per-target optimization.

### Background: Why Previous S1 vs S2 Showed No Difference

The original grid search tested four strategies:
- S1 (Joint): MultiOutputRegressor with shared hyperparams
- S2 (Separate): 3 models with shared hyperparams
- S3 (Per-participant): 5 models with shared hyperparams
- S4 (Per-participant + per-target): 15 models with shared hyperparams

**Problem**: S1 and S2 both used identical hyperparameters for all targets. MultiOutputRegressor internally creates separate models per target - functionally identical to S2. This is why results were identical (0.029338).

### Configuration

| Parameter | Value |
|-----------|-------|
| Data | 345 training shots, 5 participants |
| Features | F4 (hybrid with participant ID), 132 features |
| Model | M1 (LightGBM) |
| Preprocessing | P4 (standardized) |
| CV | GroupKFold by participant, 5 folds |
| Optuna trials | 30 per target |
| Per-participant | Yes (fallback model for held-out participant) |

### Script

```
src/per_target_experiment.py
```

Command:
```bash
uv run python src/per_target_experiment.py --n-trials 30
```

### Three Approaches Compared

1. **Baseline**: Per-player models with shared LightGBM defaults
   - n_estimators=500, learning_rate=0.02, num_leaves=20

2. **Global Tuned**: Optuna tunes hyperparams to minimize combined MSE across all 3 targets
   - Same params for angle, depth, left_right

3. **Per-Target Tuned**: Optuna tunes hyperparams independently for each target
   - Different optimal params for angle vs depth vs left_right

### Results

| Approach | Angle MSE | Depth MSE | L/R MSE | Total MSE | vs Baseline |
|----------|-----------|-----------|---------|-----------|-------------|
| Baseline (shared params) | 0.0262 | 0.0326 | 0.0176 | 0.0255 | - |
| Global tuned | 0.0233 | 0.0186 | 0.0161 | 0.0193 | +24.2% |
| **Per-target tuned** | **0.0207** | **0.0177** | **0.0154** | **0.0179** | **+29.8%** |

**Per-target tuning beats global tuning by 7.3%** (0.0179 vs 0.0193)

### Optimal Hyperparameters Per Target

| Target | n_estimators | learning_rate | Character |
|--------|-------------|---------------|-----------|
| angle | 111 | 0.0646 | Aggressive - fewer trees, higher lr |
| depth | 176 | 0.0072 | Conservative - more trees, lower lr |
| left_right | 154 | 0.0050 | Most conservative - lowest lr |

### Key Findings

1. **Per-target tuning provides significant improvement**: 29.8% over baseline, 7.3% over global tuning

2. **Each target benefits from different hyperparameters**:
   - **angle**: Aggressive learning (lr=0.065) with fewer trees (111) - complex signal needs faster adaptation
   - **depth**: Conservative (lr=0.007) with more trees (176) - weaker signal needs careful regularization
   - **left_right**: Most conservative (lr=0.005) - weakest signal, avoid overfitting

3. **Depth showed biggest per-target gain**: 45.7% improvement vs baseline (0.0326 -> 0.0177)

4. **This validates the hypothesis**: Targets ARE fundamentally different biomechanically:
   - angle: lower body mechanics at frame 153 (R2=0.45)
   - depth: left hand positioning at frame 102 (R2=0.08)
   - left_right: right finger control at frame 237 (R2=0.025)

### Conclusion

**Optimal strategy: Per-player + Per-target with independent hyperparameter tuning**

The combination of:
1. Per-player models (validated in S3 vs S1: +16% improvement)
2. Per-target hyperparameter tuning (validated here: +7.3% over global tuning)

This means the optimal approach uses **15 independently-tuned models** (5 players x 3 targets), each with hyperparameters optimized for that specific player-target combination.

### Recommended Production Configuration

For each (player, target) combination, tune hyperparameters using Optuna with:
- 30+ trials per combination
- GroupKFold CV within that player's data
- Optimize for that specific target's scaled MSE

Expected improvement over baseline S3: ~30%+ (combining per-player benefit with per-target tuning)

---

## Submission 3 - Per-Player Per-Target Model

**Date**: 2026-01-28

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | LightGBM (per-player per-target) |
| Features | F4 (hybrid with participant ID), 132 features |
| Strategy | 15 models (5 players x 3 targets) |
| Preprocessing | StandardScaler |
| Hyperparameter tuning | Optuna, 30 trials per target |

### Tuned Hyperparameters

| Target | n_estimators | learning_rate | max_depth | num_leaves |
|--------|-------------|---------------|-----------|------------|
| angle | 111 | 0.0646 | 10 | 30 |
| depth | 176 | 0.00724 | 12 | 5 |
| left_right | 154 | 0.00502 | 5 | 21 |

### Results

| Metric | Score |
|--------|-------|
| CV Score (training) | 0.0179 |
| **Leaderboard Score** | **0.010559** |

### CV Breakdown

| Target | CV MSE |
|--------|--------|
| angle | 0.0207 |
| depth | 0.0177 |
| left_right | 0.0154 |

### Script

```bash
uv run python src/create_submission.py
```

### File

`submission/submission_3.csv`

### Notes

- Leaderboard score (0.0106) better than CV score (0.0179)
- Per-target hyperparameter tuning validated as effective approach
- Each target benefits from different learning rates and tree configurations

---

## Submission 8 - Per-Participant Internal 5-Fold CV (Best Score)

**Date**: 2026-01-29

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | LightGBM |
| Features | F4 (hybrid), 126 features |
| Strategy | 15 models (5 players x 3 targets) |
| CV Method | **Internal 5-fold CV within each participant** |
| Preprocessing | StandardScaler per participant |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| num_leaves | 10 |
| learning_rate | 0.05 |
| reg_alpha | 0.5 |
| reg_lambda | 0.5 |

### Results

| Metric | Score |
|--------|-------|
| CV Score (scaled MSE) | 0.01054 |
| **Leaderboard Score** | **0.010220** |

### CV Breakdown (Raw MSE)

| Target | Raw MSE | Scaled MSE |
|--------|---------|------------|
| angle | 6.80 | 0.0076 |
| depth | 19.05 | 0.0108 |
| left_right | 13.58 | 0.0133 |

### Per-Player CV Results

| Player | Samples | angle MSE | depth MSE | left_right MSE |
|--------|---------|-----------|-----------|----------------|
| 1 | 70 | 1.65 | 12.67 | 16.28 |
| 2 | 66 | 4.99 | 15.17 | 17.12 |
| 3 | 68 | 2.94 | 3.77 | 6.00 |
| 4 | 67 | 6.15 | 18.78 | 14.05 |
| 5 | 74 | 17.40 | 42.84 | 14.43 |

### Script

```bash
uv run python src/create_submission_0104.py
```

### File

`submission/submission_8.csv`

### Key Differences from Submission 3

| Aspect | Submission 3 | Submission 8 |
|--------|--------------|--------------|
| CV Method | Leave-one-participant-out | Internal 5-fold within participant |
| Hyperparameters | Optuna-tuned per target | Simple fixed params |
| n_estimators | 111-176 | 100 |
| learning_rate | 0.005-0.065 | 0.05 |
| Leaderboard | 0.010559 | **0.010220** |

### Notes

- **Best leaderboard score achieved**: 0.010220
- Simpler hyperparameters (fixed across all targets) outperformed complex per-target tuning
- Internal 5-fold CV within each participant is more representative of test set distribution
- Player 5 shows highest error across all targets (potential area for improvement)

---

## 2026-01-29 - Advanced Feature Engineering and Ensemble Optimization

### Objective

Beat the winners' score of 0.007 MSE.

### Key Insight: Player 5 Variance Analysis

Player 5 has significantly higher variance in all targets:

| Target | Player 5 Std | Other Players Std Range | Ratio |
|--------|--------------|------------------------|-------|
| angle | 4.10 | 1.30-2.70 | 1.5-3x higher |
| depth | 8.16 | 2.32-4.85 | 1.7-3.5x higher |
| left_right | 4.16 | 2.85-4.13 | Similar |

Player 5 is inherently more inconsistent in their shooting, making prediction harder.

### New Feature Engineering

1. **Frame-specific features** based on research findings:
   - Frame 153 for ANGLE (ankle/knee z, R2=0.45)
   - Frame 102 for DEPTH (hand positions, R2=0.08)
   - Frame 237 for LEFT_RIGHT (finger positions, R2=0.025)

2. **Advanced features** (src/advanced_features.py):
   - 62 angle-critical features
   - 48 depth-critical features
   - 47 left_right-critical features
   - 10 release features
   - 64 phase features
   - Total: 242 new features

3. Combined with hybrid features: 368 total features

### Submission Results

| Submission | Model | CV Score | LB Score | Notes |
|------------|-------|----------|----------|-------|
| 9 | Ensemble (LGB+Cat+XGB+Ridge) | 0.008441 | **0.009109** | First improved ensemble |
| 10 | Optuna-tuned ensemble | 0.007919 | TBD | Per-player per-target tuning |
| 11 | Ultra-optimized | **0.007767** | TBD | Best CV, Player 5 special handling |
| 12 | Robust bagging (5 seeds) | 0.008198 | TBD | More variance reduction |
| 13 | Blend (10+11+12) | - | TBD | Weighted by inverse CV |
| 14 | Player 5 mean-blend | 0.008533 | TBD | Conservative for high-variance player |
| 15 | Blend (heavy on 11) | - | TBD | 20% sub9 + 30% sub10 + 50% sub11 |
| 16 | Blend (heavy on 9) | - | TBD | 50% sub9 + 25% sub10 + 25% sub11 |
| 17 | Simple Ridge | 0.010923 | TBD | Baseline comparison |

### Best Configuration (Submission 11)

**Model**: Ultra-optimized ensemble

**Strategy**:
- Per-player per-target models (15 total)
- 4 base models: LightGBM, CatBoost, XGBoost, Ridge
- Optuna tuning: 50 trials for Player 5, 25 for others
- More regularization for Player 5
- Target-specific feature selection (80-120 features per target)

**Hyperparameter strategy**:
- Players 1-4: Standard optimization
- Player 5: Smaller trees, higher regularization, more conservative

**CV Breakdown**:

| Player | angle MSE | depth MSE | left_right MSE |
|--------|-----------|-----------|----------------|
| 1 | 1.58 | 7.33 | 9.89 |
| 2 | 4.14 | 9.91 | 11.70 |
| 3 | 2.57 | 2.96 | 5.39 |
| 4 | 4.96 | 10.75 | 11.12 |
| 5 | 14.07 | 28.21 | 13.95 |

**Overall CV**: 0.007767

### Progress Summary

| Metric | Previous Best | New Best | Improvement |
|--------|---------------|----------|-------------|
| CV Score | 0.01054 | 0.007767 | 26.3% |
| LB Score | 0.010220 | 0.009109 | 10.9% |

### Recommendations

Submit in order:
1. **submission_11.csv** - Best CV (0.007767)
2. **submission_15.csv** - Blend with proven LB performer
3. **submission_10.csv** - Second best CV (0.007919)
4. **submission_13.csv** - Conservative blend

### Files Created

- `src/advanced_features.py` - Frame-specific feature engineering
- `src/ensemble_submission.py` - Multi-model ensemble
- `src/optimized_ensemble.py` - Optuna-tuned ensemble
- `src/ultra_optimized.py` - Best performing model
- `src/robust_ensemble.py` - Bagging with multiple seeds
- `src/final_submission.py` - Player 5 mean-blending
- `src/blend_submissions.py` - Submission blending
- `src/simple_ridge.py` - Ridge baseline
