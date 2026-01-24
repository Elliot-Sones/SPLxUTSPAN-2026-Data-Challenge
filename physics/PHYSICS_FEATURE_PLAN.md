# Plan: Physics-Based Feature Engineering

## Problem Analysis

**Current state**: 3365 features, 92% are generic stats (f0_mean, f1_std, etc.)
**Result**: Scaled MSE ~0.035, leader is 0.008381 (4x gap)
**Root cause**: Features don't capture what physics says matters

---

## Physics of Shot Prediction

Once the ball leaves the hand, its trajectory is **completely determined** by 6 quantities at release:
- Position: (x, y, z) - where the ball starts
- Velocity: (vx, vy, vz) - direction and speed

### What Predicts Each Target

| Target | Physics | Key Feature |
|--------|---------|-------------|
| **angle** | Entry angle = f(release arc) | `wrist_vz / wrist_vy` at release |
| **depth** | Short/long = forward velocity | `wrist_vy` at release |
| **left_right** | Lateral = sideways velocity | `wrist_vx` at release + elbow alignment |

---

## Data Quality Assessment

- **Position noise**: 0.026 units (excellent)
- **Signal-to-noise ratio**: 138:1
- **Release frame detection**: 4:1 SNR, +/-1-2 frames accuracy
- **Conclusion**: Data is clean, minimal preprocessing needed

---

## Empirical Evidence: What Features Actually Matter

Analysis of current model feature importance (LightGBM vs XGBoost):

### Top Features by Target (LightGBM)

| Target | #1 Feature | Type | Importance |
|--------|-----------|------|------------|
| **DEPTH** | `knee_extension_rate_mean` | Physics | Highest |
| **ANGLE** | `right_knee_angle_min` | Physics | Highest |
| **LEFT_RIGHT** | Generic z-coords | Stats | Mixed |

### Physics vs Generic Feature Importance

| Model | Generic Stats | Physics | Velocity | Phase |
|-------|--------------|---------|----------|-------|
| XGBoost | 73-77% | 4-6% | 9-16% | 3-11% |
| LightGBM | 53-65% | 9-14% | 14-17% | 10-15% |

### Key Insights

1. **Physics features work**: `knee_extension_rate_mean` is the single most important feature for depth prediction
2. **Generic features are redundant**: Top generic features (f32, f44, etc.) are just z-coordinates of key joints
3. **LightGBM uses physics better**: 14% physics importance vs XGBoost's 6%
4. **Low cross-model overlap**: Only 1-3 features in common top-30, suggesting ensemble value

---

## Strategy: Physics-First Feature Engineering

### Core Principle
Replace 3000+ noisy generic features with ~50 physics-based features that directly relate to shot outcome.

### Feature Categories

**Category 1: Release Velocity Vector (6 features)**
The most important features - directly determine trajectory.
```
wrist_vx, wrist_vy, wrist_vz at release frame
velocity_magnitude = sqrt(vx^2 + vy^2 + vz^2)
elevation_angle = atan2(vz, vy)  # predicts entry angle
azimuth_angle = atan2(vx, vy)    # predicts left/right
```

**Category 2: Release Position (6 features)**
Where the ball starts its trajectory.
```
wrist_x, wrist_y, wrist_z at release
release_height_relative = wrist_z - hip_z
lateral_offset = wrist_x - shoulder_x
forward_position = wrist_y
```

**Category 3: Arm Alignment at Release (6 features)**
Body mechanics that affect velocity direction.
```
elbow_alignment = elbow_x - wrist_x  # should be ~0 for straight shot
shoulder_elbow_wrist_angle  # arm extension
wrist_snap_angle = angle(elbow, wrist, finger)
shoulder_rotation = angle of shoulder line vs forward
arm_plane_angle = is arm moving in vertical plane?
elbow_height_relative = elbow_z - shoulder_z
```

**Category 4: Velocity Derivatives (4 features)**
Timing and consistency indicators.
```
release_frame  # when release happens
acceleration_at_release  # is velocity still increasing?
jerk_at_release  # smoothness
time_to_peak_velocity
```

**Category 5: Pre-Release Mechanics (8 features)**
Upstream factors that affect release quality.
```
knee_bend_depth = min(knee_angle) during propulsion
knee_extension_rate = d(knee_angle)/dt max
set_point_height = wrist_z at pause before release
hip_vertical_velocity = upward momentum
shoulder_elevation_rate
guide_hand_interference = off_wrist_vx at release
balance = hip_lateral_movement during shot
consistency = variance of key positions
```

**Category 6: Participant Features (5 features)**
Account for individual shooting styles.
```
participant_id (one-hot: 5 features)
```

**TOTAL: ~35-40 physics-based features**

---

## Implementation Plan: Full Factorial Experiment

### Experimental Design

Test ALL combinations of (Features x Algorithms x Training Approach) and let data determine best configuration.

**Independent Variables:**

| Dimension | Options | Count |
|-----------|---------|-------|
| Features | baseline (414), physics (~40), hybrid (~130), stats_full (~1200) | 4 |
| Algorithm | LightGBM, XGBoost, CatBoost, Ridge, RandomForest | 5 |
| Training | Shared model, Per-participant | 2 |
| Target | angle, depth, left_right | 3 |

**Total Experiments:** 4 x 5 x 2 x 3 = **120 combinations**

**Dependent Variable:** Scaled MSE (5-fold CV)

### Key Insight: Shared vs Per-Participant

Initial experiment showed per-participant models 56.5% better overall:
- Per-participant scaled MSE: 0.0104
- Shared model scaled MSE: 0.0238

However, this may not hold for all combinations:
- **left_right** has NO participant effect (ANOVA p=0.47) - shared may be better
- Complex features with ~70 samples may overfit per-participant
- Simpler algorithms (Ridge) may benefit more from shared (more data)

**Decision: Test both training approaches for every combination.**

### Experiments to Run

**Phase 1: Feature x Algorithm (Shared Model)**

| # | Features | Algorithm | Training | CV Method |
|---|----------|-----------|----------|-----------|
| 1-20 | All 4 | All 5 | Shared | Leave-One-Participant-Out |

**Phase 2: Feature x Algorithm (Per-Participant)**

| # | Features | Algorithm | Training | CV Method |
|---|----------|-----------|----------|-----------|
| 21-40 | All 4 | All 5 | Per-participant | 5-fold within participant |

**Phase 3: Compare and Select**

For each (Feature, Algorithm) pair:
- Compare shared vs per-participant for each target
- Select winner per target (may differ!)

### Expected Outcome

Final model may be hybrid:
```
angle:      per-participant + hybrid + LightGBM
depth:      per-participant + physics + XGBoost
left_right: shared + baseline + Ridge
```

Each target gets its own optimal (features, algorithm, training) configuration.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/physics_features.py` | Pure physics feature extraction |
| `src/hybrid_features.py` | Physics + selected stats |
| `src/run_ablation.py` | Run all experiments, compare results |
| `src/train_separate.py` | Train separate model per target |

---

## Verification

1. All feature sets produce no NaN values
2. Release velocity in range 7-15 units/sec
3. Each experiment runs successfully on GPU
4. Results table shows clear winner or close competitors
5. Winner beats current 0.035 scaled MSE significantly

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Best experiment scaled MSE | < 0.020 |
| Improvement over current | > 40% |
| Feature importance | Physics features in top 10 |

---

## Decision Tree After Ablation

```
If physics-only wins:
  -> Use ~40 features, simpler model
  -> Focus on feature refinement

If hybrid wins:
  -> Use ~80 features
  -> Consider feature selection to reduce further

If current wins:
  -> Physics hypothesis wrong
  -> Investigate why generic stats work
```

---

## Step 1: Create Feature Extractors

**File: `src/physics_features.py`**
- Pure physics-based features (~40)
- Release velocity vector (6)
- Release position (6)
- Arm alignment (6)
- Velocity derivatives (4)
- Pre-release mechanics (8)
- Participant one-hot (5)

**File: `src/hybrid_features.py`**
- Physics features + top z-coordinate stats
- wrist_z, elbow_z, knee_z, hip_z, shoulder_z
- Mean, std, min, max, range for each
- Total ~80 features

### Step 2: Add Smoothing Option

```python
def smooth_timeseries(data, window=5):
    from scipy.signal import savgol_filter
    return savgol_filter(data, window, polyorder=2, axis=0)
```

Apply before velocity computation when smoothing=True.

### Step 3: Train Separate Models Per Target

For each target (angle, depth, left_right):
- Train independent LightGBM model
- Optimize hyperparameters separately
- Each target may need different features

### Step 4: Run All Experiments

On GPU (vast.ai):
```bash
python src/run_ablation.py --experiments all
```

Output: Table comparing all approaches on CV MSE.

### Step 5: Select Best Approach

Pick the experiment with lowest scaled MSE.
If close, prefer simpler (fewer features).
