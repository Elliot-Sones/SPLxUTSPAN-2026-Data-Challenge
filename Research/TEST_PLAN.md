# Comprehensive Experimental Test Plan

**Objective**: Systematically find the best (algorithm, features, training approach) combination for each target through full factorial experimentation.

**Reference baseline**: 0.010 MSE achieved with mean + last features + RandomForest
**Leaderboard leader**: 0.008381 MSE
**Target**: < 0.010 scaled MSE

---

## Key Principles

1. **Compute is NOT a constraint** - 1000 experiments take ~2 hours on CPU
2. **Test all 3 targets independently** - Different features may matter for angle vs depth vs left_right
3. **Measure variance, not just mean** - Report mean +/- std for all results
4. **Statistical significance** - Only trust differences > 2x standard deviation
5. **Track interactions** - Algorithm x Features x Training x Target interactions may exist
6. **Test BOTH training approaches** - Per-participant showed 56.5% improvement overall, but left_right has NO participant effect (ANOVA p=0.47) so shared may be better for that target

---

## CRITICAL FINDING: Per-Participant vs Shared Models

### Experiment: Shared Model vs Per-Participant Models

**Hypothesis**: A shared model learns patterns from one participant that incorrectly transfer to others. Per-participant models avoid this cross-contamination.

**Method**:
- Shared model: Train on 4 participants, test on held-out participant (Leave-One-Participant-Out CV)
- Per-participant: Train on ~70 samples from one participant, validate with 5-fold CV within that participant

**IMPORTANT UPDATE**: Initial results showed per-participant 56.5% better overall, BUT:
- left_right has NO participant effect (ANOVA p=0.47) - shared may be better
- Complex features with ~70 samples may overfit per-participant
- Simpler algorithms (Ridge) may benefit more from shared (more data)

**Decision: TEST BOTH APPROACHES FOR EVERY COMBINATION**

### Results

**Per-Target MSE Comparison:**

| Target     | Shared Model | Per-Participant | Improvement |
|------------|--------------|-----------------|-------------|
| angle      |      18.72   |          6.45   | **+65.6%** |
| depth      |      53.79   |         17.76   | **+67.0%** |
| left_right |      20.63   |         14.21   | **+31.1%** |

**Scaled MSE:**

| Approach | Scaled MSE | vs Leaderboard (0.008381) |
|----------|------------|---------------------------|
| Shared model | 0.0238 | 2.8x worse |
| Per-participant | **0.0104** | 1.2x worse |

**Per-participant models are 56.5% better overall!**

### Per-Participant Breakdown

**Shared Model (trained on OTHER participants):**

| PID | angle MSE | depth MSE | left_right MSE |
|-----|-----------|-----------|----------------|
| 1   |    5.23   |   83.95   |   23.77 |
| 2   |    6.76   |   27.55   |   17.46 |
| 3   |   19.79   |   32.90   |   27.79 |
| 4   |   30.30   |   36.06   |   15.21 |
| 5   |   31.53   |   88.51   |   18.93 |

**Per-Participant Model (trained on SAME participant):**

| PID | angle MSE | depth MSE | left_right MSE |
|-----|-----------|-----------|----------------|
| 1   |    1.78   |   12.58   |   16.26 |
| 2   |    4.70   |   15.52   |   17.81 |
| 3   |    3.01   |    3.86   |    5.89 |
| 4   |    5.87   |   17.29   |   14.18 |
| 5   |   16.87   |   39.54   |   16.87 |

### Why This Works

1. **Each participant has a unique shooting style**:
   - P4 averages 52.3° angle, P5 averages 40.6° (12° difference)
   - A shared model tries to learn both, causing confusion

2. **Simpler learning task**:
   - Shared: 345 samples to learn 5 different styles
   - Per-participant: 70 samples to learn 1 style
   - Learning 1 style with 70 samples > learning 5 styles with 345 samples

3. **No cross-contamination**:
   - P5 has 2x variance in depth compared to others
   - Shared model might apply P5's high-variance patterns to P2
   - Per-participant model avoids this entirely

4. **Test set has same participants**:
   - We know participant ID for every test shot
   - Can route each test shot to the correct participant's model
   - No risk of overfitting - same participants in train and test

### Implementation

**Architecture**: 15 models total (5 participants x 3 targets)

```python
models = {}
for participant_id in [1, 2, 3, 4, 5]:
    for target in ['angle', 'depth', 'left_right']:
        # Train on this participant's data only
        mask = participant_ids == participant_id
        X_pid = X[mask]
        y_pid = y[mask, target_idx]

        model = LGBMRegressor(n_estimators=100, num_leaves=10,
                              learning_rate=0.05, reg_alpha=0.5, reg_lambda=0.5)
        model.fit(X_pid, y_pid)
        models[(participant_id, target)] = model

# Prediction: route to correct participant's model
def predict(X_test, participant_ids_test):
    predictions = []
    for i, pid in enumerate(participant_ids_test):
        pred = [models[(pid, t)].predict(X_test[i:i+1])[0] for t in targets]
        predictions.append(pred)
    return np.array(predictions)
```

### Decision

**TEST BOTH TRAINING APPROACHES FOR ALL EXPERIMENTS**

Each target may have a different optimal training approach:
- **angle**: Likely per-participant (strong participant effect)
- **depth**: Likely per-participant (strong participant effect)
- **left_right**: May favor shared model (NO participant effect, p=0.47)

All subsequent phases should test BOTH approaches:
- Phase 1: Algorithm comparison with both shared and per-participant
- Phase 2: Feature comparison with both shared and per-participant
- Phase 3: Select best (features, algorithm, training) per target

---

## Phase 0: Establish Experimental Framework

### 0.1 Baseline Variance Measurement

Before comparing anything, measure inherent variance in our evaluation:

```python
# Run same model 10x with different seeds
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import numpy as np

baseline_scores = {'angle': [], 'depth': [], 'left_right': [], 'total': []}
logo = LeaveOneGroupOut()

for seed in range(10):
    model = LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1)
    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        scores = cross_val_score(model, X_baseline, y[:, t_idx],
                                 cv=logo, groups=participant_ids,
                                 scoring='neg_mean_squared_error')
        baseline_scores[target].append(-scores.mean())
    # Total
    total_score = np.mean([baseline_scores[t][-1] for t in ['angle', 'depth', 'left_right']])
    baseline_scores['total'].append(total_score)

for target, scores in baseline_scores.items():
    print(f"{target}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
```

**Results:**

| Target | Mean MSE | Std | Min Detectable Difference (2*std) |
|--------|----------|-----|-----------------------------------|
| angle | | | |
| depth | | | |
| left_right | | | |
| **total** | | | |

**Interpretation**: Any difference smaller than 2*std is likely noise.

---

### 0.2 Reference Baseline Implementation

Implement the 0.010 MSE baseline for comparison:

```python
# Baseline features: mean + last value per column
def extract_baseline_features(df, feature_cols):
    """414 features: mean + last for each of 207 columns"""
    features = []
    for idx, row in df.iterrows():
        row_feats = []
        for col in feature_cols:
            arr = np.array(row[col]).flatten()
            row_feats.append(np.mean(arr))  # mean
            row_feats.append(arr[-1])       # last value
        features.append(row_feats)
    return np.array(features)
```

| Target | Baseline MSE (mean+last, RF) |
|--------|------------------------------|
| angle | |
| depth | |
| left_right | |
| **total** | |

---

## Phase 1: Full Factorial Experiment Design

**Goal**: Test ALL combinations of (Features x Algorithms x Training Approach) and let data determine best configuration.

### 1.1 Experimental Design

**Independent Variables:**

| Dimension | Options | Count |
|-----------|---------|-------|
| Features | baseline (414), physics (~40), hybrid (~130), stats_full (~1200) | 4 |
| Algorithm | LightGBM, XGBoost, CatBoost, Ridge, RandomForest | 5 |
| Training | Shared model, Per-participant | 2 |
| Target | angle, depth, left_right | 3 |

**Total Experiments:** 4 x 5 x 2 x 3 = **120 combinations**

**Dependent Variable:** Scaled MSE (5-fold CV)

### 1.2 CV Methods by Training Approach

- **Shared model**: Leave-One-Participant-Out (5 folds)
- **Per-participant**: 5-fold CV within each participant, averaged across participants

### 1.3 Full Factorial Results Table

| Features | Algorithm | Training | angle MSE | depth MSE | l/r MSE | Total Scaled |
|----------|-----------|----------|-----------|-----------|---------|--------------|
| baseline | LightGBM | shared | | | | |
| baseline | LightGBM | per-part | | | | |
| baseline | XGBoost | shared | | | | |
| baseline | XGBoost | per-part | | | | |
| baseline | CatBoost | shared | | | | |
| baseline | CatBoost | per-part | | | | |
| baseline | Ridge | shared | | | | |
| baseline | Ridge | per-part | | | | |
| baseline | RandomForest | shared | | | | |
| baseline | RandomForest | per-part | | | | |
| physics | LightGBM | shared | | | | |
| physics | LightGBM | per-part | | | | |
| physics | XGBoost | shared | | | | |
| physics | XGBoost | per-part | | | | |
| physics | CatBoost | shared | | | | |
| physics | CatBoost | per-part | | | | |
| physics | Ridge | shared | | | | |
| physics | Ridge | per-part | | | | |
| physics | RandomForest | shared | | | | |
| physics | RandomForest | per-part | | | | |
| hybrid | LightGBM | shared | | | | |
| hybrid | LightGBM | per-part | | | | |
| hybrid | XGBoost | shared | | | | |
| hybrid | XGBoost | per-part | | | | |
| hybrid | CatBoost | shared | | | | |
| hybrid | CatBoost | per-part | | | | |
| hybrid | Ridge | shared | | | | |
| hybrid | Ridge | per-part | | | | |
| hybrid | RandomForest | shared | | | | |
| hybrid | RandomForest | per-part | | | | |
| stats_full | LightGBM | shared | | | | |
| stats_full | LightGBM | per-part | | | | |
| stats_full | XGBoost | shared | | | | |
| stats_full | XGBoost | per-part | | | | |
| stats_full | CatBoost | shared | | | | |
| stats_full | CatBoost | per-part | | | | |
| stats_full | Ridge | shared | | | | |
| stats_full | Ridge | per-part | | | | |
| stats_full | RandomForest | shared | | | | |
| stats_full | RandomForest | per-part | | | | |

### 1.4 Per-Target Winner Selection

Each target gets its own optimal (features, algorithm, training) configuration:

| Target | Best Features | Best Algorithm | Best Training | Scaled MSE |
|--------|--------------|----------------|---------------|------------|
| angle | | | | |
| depth | | | | |
| left_right | | | | |

**Final hybrid model may look like:**
```
angle:      per-participant + hybrid + LightGBM
depth:      per-participant + physics + XGBoost
left_right: shared + baseline + Ridge
```

### 1.5 Statistical Significance Tests

For each target, compare best vs second best configuration:

| Target | Best Config | Second Best | p-value | Significant? |
|--------|-------------|-------------|---------|--------------|
| angle | | | | |
| depth | | | | |
| left_right | | | | |

### 1.6 Key Interactions to Analyze

1. **Training x Target**: Does left_right actually prefer shared? (hypothesis: yes)
2. **Features x Algorithm**: Do physics features work better with certain algorithms?
3. **Features x Training**: Do complex features overfit with per-participant (few samples)?

**Conclusion Phase 1**: Select best configuration per target for final model

---

## Phase 2: Feature Set Definitions

**Goal**: Define and implement all feature sets for factorial experiment.

### 2.1 Feature Set Definitions

| Feature Set | # Features | Description |
|-------------|------------|-------------|
| baseline | 414 | mean + last per column |
| physics | ~40 | release velocity, position, arm alignment, velocity derivatives, pre-release mechanics |
| hybrid | ~130 | physics + z-coord stats for key joints |
| stats_full | ~1200 | mean, std, min, max, median, last per column |

### 2.2 Physics Feature Categories (~40 features)

| Category | Features | Count |
|----------|----------|-------|
| Release Velocity Vector | wrist_vx, wrist_vy, wrist_vz, velocity_magnitude, elevation_angle, azimuth_angle | 6 |
| Release Position | wrist_x, wrist_y, wrist_z, release_height_relative, lateral_offset, forward_position | 6 |
| Arm Alignment | elbow_alignment, arm_extension_angle, wrist_snap_angle, shoulder_rotation, arm_plane_angle, elbow_height_relative | 6 |
| Velocity Derivatives | acceleration_at_release, jerk_at_release, time_to_peak_velocity, peak_velocity_magnitude | 4 |
| Pre-Release Mechanics | knee_bend_depth, knee_extension_rate_max, set_point_height, hip_vertical_velocity_max, shoulder_elevation_rate, guide_hand_vx, hip_lateral_range, set_position_stability | 8 |
| Participant | participant_id, participant_1-5 (one-hot) | 6 |

### 2.3 Hybrid Feature Categories (~130 features)

Physics features PLUS:

| Category | Joints | Stats | Count |
|----------|--------|-------|-------|
| Z-coord stats | right_wrist, right_elbow, right_shoulder, right_knee, right_hip, mid_hip, left_wrist, neck | mean, std, min, max, range, q25, q75, energy | 64 |
| Velocity stats | right_wrist, right_elbow, right_knee | mean, std, max, min, range, max_time, phase means | 30 |
| Angle stats | right_elbow, right_knee, right_shoulder | mean, std, min, max, range | 15 |

### 2.4 Feature Extraction Commands

```bash
# Physics features
uv run python src/run_ablation.py --experiments physics --model lightgbm

# Hybrid features
uv run python src/run_ablation.py --experiments hybrid --model lightgbm

# Baseline features
uv run python src/run_ablation.py --experiments baseline --model lightgbm
```

### 2.5 Feature Quality Verification

Before running experiments, verify:
- [ ] No NaN values in any feature set
- [ ] Release velocity magnitude in range 7-15 units/sec
- [ ] All features have reasonable variance (not constant)

**Note**: Phase 1 full factorial already tests all feature sets with both training approaches

---

## Phase 3: Fine-Grained Feature Ablation

**Goal**: Understand exactly which features help, which hurt.

### 3.1 Leave-One-Feature-Group-Out

For best feature set, remove one group at a time:

```python
feature_groups = {
    'position_mean': [...],
    'position_last': [...],
    'velocity': [...],
    'acceleration': [...],
    'angles': [...],
    # etc.
}

for group_name, indices in feature_groups.items():
    X_ablated = np.delete(X_best, indices, axis=1)
    score = cross_val_score(model, X_ablated, y, cv=logo).mean()
    impact = score - best_score  # positive = removing hurts (feature is useful)
```

| Feature Group Removed | angle Impact | depth Impact | left_right Impact |
|----------------------|--------------|--------------|-------------------|
| position_mean | | | |
| position_last | | | |
| velocity | | | |
| acceleration | | | |
| angles | | | |
| ... | | | |

### 3.2 Add-One-Feature-Group

Start with minimal features, add groups:

| Feature Group Added | angle MSE | depth MSE | left_right MSE | Cumulative Benefit |
|--------------------|-----------|-----------|----------------|-------------------|
| release_position | | | | |
| + velocity | | | | |
| + angles | | | | |
| + statistics | | | | |
| ... | | | | |

### 3.3 Per-Target Feature Importance (Permutation)

```python
from sklearn.inspection import permutation_importance

for target in ['angle', 'depth', 'left_right']:
    model.fit(X_best, y[target])
    result = permutation_importance(model, X_val, y_val[target], n_repeats=30)
    # Top 10 features for this target
```

**Top 10 Features per Target:**

| Rank | angle | depth | left_right |
|------|-------|-------|------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

### 3.4 Feature Overlap Analysis

| Feature | In angle top 20? | In depth top 20? | In left_right top 20? |
|---------|-----------------|-----------------|----------------------|
| | | | |
| | | | |
| | | | |

**Universal features** (help all targets):
**Target-specific features**:
- angle only:
- depth only:
- left_right only:

---

## Phase 4: Frame Importance Analysis

**Goal**: Determine which frames (time points) matter most.

### 4.1 Sliding Window Analysis

```python
window_size = 30
stride = 10

for target in ['angle', 'depth', 'left_right']:
    window_scores = []
    for start in range(0, 240 - window_size, stride):
        X_window = extract_features(shots, frame_range=(start, start + window_size))
        score = cross_val_score(model, X_window, y[target], cv=logo).mean()
        window_scores.append((start, score))
```

**Results:**

| Frame Range | angle MSE | depth MSE | left_right MSE |
|-------------|-----------|-----------|----------------|
| 0-30 | | | |
| 30-60 | | | |
| 60-90 | | | |
| 90-120 | | | |
| 120-150 | | | |
| 150-180 | | | |
| 180-210 | | | |
| 210-240 | | | |

**Best window per target:**

| Target | Best Frame Range | MSE | Interpretation |
|--------|-----------------|-----|----------------|
| angle | | | |
| depth | | | |
| left_right | | | |

### 4.2 Release Frame Detection Comparison

Test different release detection methods:

| Detection Method | angle MSE | depth MSE | left_right MSE |
|-----------------|-----------|-----------|----------------|
| max wrist_z | | | |
| max velocity | | | |
| max acceleration | | | |
| fixed frame 180 | | | |

---

## Phase 5: Joint/Coordinate Importance

**Goal**: Determine which body parts and coordinates matter.

### 5.1 Per-Joint Ablation

```python
joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip', ...]

for joint in joints:
    X_joint_only = extract_features_single_joint(joint)
    for target in targets:
        score = cross_val_score(model, X_joint_only, y[target], cv=logo).mean()
```

| Joint | angle MSE | depth MSE | left_right MSE |
|-------|-----------|-----------|----------------|
| right_wrist | | | |
| right_elbow | | | |
| right_shoulder | | | |
| right_hip | | | |
| right_knee | | | |
| neck | | | |
| mid_hip | | | |
| right_fingers | | | |
| left_wrist (guide) | | | |

### 5.2 Per-Coordinate Ablation

| Coordinate | angle MSE | depth MSE | left_right MSE |
|------------|-----------|-----------|----------------|
| x only | | | |
| y only | | | |
| z only | | | |
| x + z | | | |
| y + z | | | |

**Hypothesis**: z (vertical) matters most for angle/depth, x matters for left_right

---

## Phase 6: Deep Learning Comparison

**Goal**: Test if CNN + Attention can beat engineered features.

### 6.1 Raw Time Series Baseline

| Model | angle MSE | depth MSE | left_right MSE | Total | Train Time |
|-------|-----------|-----------|----------------|-------|------------|
| LSTM (no attention) | | | | | |
| CNN (no attention) | | | | | |
| CNN + Frame Attention | | | | | |
| Transformer | | | | | |

### 6.2 Attention Weight Visualization

If CNN + Attention is competitive, visualize which frames it attends to:

| Target | Peak Attention Frames | Matches Phase 4 findings? |
|--------|----------------------|---------------------------|
| angle | | |
| depth | | |
| left_right | | |

### 6.3 Hybrid: Engineered Features + Learned Features

```python
# Extract learned features from attention model bottleneck
learned_features = attention_model.get_bottleneck(X_raw)

# Combine with engineered features
X_hybrid = np.hstack([X_engineered, learned_features])
```

| Approach | Total MSE |
|----------|-----------|
| Engineered only | |
| Learned only | |
| Hybrid | |

---

## Phase 7: Final Model Selection

### 7.1 Summary Table

| Approach | angle MSE | depth MSE | left_right MSE | Total Scaled MSE |
|----------|-----------|-----------|----------------|------------------|
| Baseline (mean+last, RF, shared) | | | | 0.010 |
| Best single config (all targets) | | | | |
| Per-target optimized | | | | |
| Ensemble (top 3 configs averaged) | | | | |

### 7.2 Selected Configuration Per Target

**For angle:**
- Features:
- Algorithm:
- Training: shared / per-participant
- Scaled MSE:

**For depth:**
- Features:
- Algorithm:
- Training: shared / per-participant
- Scaled MSE:

**For left_right:**
- Features:
- Algorithm:
- Training: shared / per-participant
- Scaled MSE:

### 7.3 Final Model Architecture

```python
# Route each target to its optimal configuration
class HybridTargetModel:
    def __init__(self):
        self.configs = {
            'angle': {'features': '...', 'algorithm': '...', 'training': '...'},
            'depth': {'features': '...', 'algorithm': '...', 'training': '...'},
            'left_right': {'features': '...', 'algorithm': '...', 'training': '...'},
        }
        self.models = {}  # Loaded based on configs

    def predict(self, X_raw, participant_id):
        predictions = {}
        for target, config in self.configs.items():
            X = extract_features(X_raw, config['features'])
            if config['training'] == 'per-participant':
                model = self.models[(target, participant_id)]
            else:
                model = self.models[(target, 'shared')]
            predictions[target] = model.predict(X)
        return predictions
```

### 7.4 Leaderboard Submissions

| Version | CV MSE | LB MSE | Gap | Notes |
|---------|--------|--------|-----|-------|
| v1 baseline | | | | mean+last, RF, shared |
| v2 per-part | | | | per-participant models |
| v3 physics | | | | physics features |
| v4 hybrid | | | | hybrid features |
| v5 factorial best | | | | best from full factorial |
| v6 per-target opt | | | | different config per target |

---

## Experiment Log

Track ALL experiments for reproducibility:

| # | Date | Feature Set | Algorithm | Target | CV MSE | Std | Notes |
|---|------|-------------|-----------|--------|--------|-----|-------|
| 1 | | baseline | RF | all | | | reference |
| 2 | | | | | | | |
| 3 | | | | | | | |

---

## ORIGINAL PHASE 1 (Renamed to Legacy)

## Legacy Phase 1: Feature Set Comparison (CV Only)

Compare different feature sets using 5-fold GroupKFold cross-validation on training data.

### Test 1.1: Physics-Only Features (Unsmoothed)

**Command**:
```bash
uv run python src/run_ablation.py --experiments physics --model lightgbm
```

**Features**: ~38 physics-based features
**Expected**: Establish physics-only baseline

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Raw MSE | |
| Angle MSE | |
| Depth MSE | |
| Left/Right MSE | |
| Feature extraction time | |
| Training time | |

**Top 5 features (angle)**:
1.
2.
3.
4.
5.

**Top 5 features (depth)**:
1.
2.
3.
4.
5.

**Top 5 features (left_right)**:
1.
2.
3.
4.
5.

---

### Test 1.2: Physics-Only Features (Smoothed)

**Command**:
```bash
uv run python src/run_ablation.py --experiments physics --model lightgbm
```

(The ablation script runs both smoothed and unsmoothed)

**Features**: ~38 physics-based features with Savitzky-Golay smoothing

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Raw MSE | |
| Angle MSE | |
| Depth MSE | |
| Left/Right MSE | |

**Smoothing impact**: Better / Worse / No difference

---

### Test 1.3: Hybrid Features (Unsmoothed)

**Command**:
```bash
uv run python src/run_ablation.py --experiments hybrid --model lightgbm
```

**Features**: ~132 features (physics + z-coord stats)

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Raw MSE | |
| Angle MSE | |
| Depth MSE | |
| Left/Right MSE | |
| Feature extraction time | |
| Training time | |

**Top 5 features (angle)**:
1.
2.
3.
4.
5.

**Top 5 features (depth)**:
1.
2.
3.
4.
5.

**Top 5 features (left_right)**:
1.
2.
3.
4.
5.

---

### Test 1.4: Hybrid Features (Smoothed)

**Command**: (same as 1.3, runs both)

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Raw MSE | |
| Angle MSE | |
| Depth MSE | |
| Left/Right MSE | |

**Smoothing impact**: Better / Worse / No difference

---

### Test 1.5: Baseline (Current 3365 Features)

**Command**:
```bash
uv run python src/run_ablation.py --experiments baseline --model lightgbm
```

**Features**: 3365 generic statistical features

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Raw MSE | |
| Angle MSE | |
| Depth MSE | |
| Left/Right MSE | |
| Feature extraction time | |
| Training time | |

---

### Phase 1 Summary Table

| Experiment | Features | Scaled MSE | vs Baseline |
|------------|----------|------------|-------------|
| A1: Physics unsmoothed | 38 | | |
| A2: Physics smoothed | 38 | | |
| B1: Hybrid unsmoothed | 132 | | |
| B2: Hybrid smoothed | 132 | | |
| C: Baseline | 3365 | | |

**Best approach from Phase 1**:

**Decision**: Proceed with _________________ for Phase 2

---

## Phase 2: Per-Target Feature Analysis

Understand which features matter for each target using permutation importance.

### Test 2.1: Feature Importance for ANGLE

**Command**:
```bash
uv run python -c "
from sklearn.inspection import permutation_importance
# ... (see full script in commands reference)
"
```

**Top 10 Features for ANGLE** (by permutation importance):

| Rank | Feature | Importance | Std |
|------|---------|------------|-----|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Physics interpretation**:

---

### Test 2.2: Feature Importance for DEPTH

**Top 10 Features for DEPTH**:

| Rank | Feature | Importance | Std |
|------|---------|------------|-----|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Physics interpretation**:

---

### Test 2.3: Feature Importance for LEFT_RIGHT

**Top 10 Features for LEFT_RIGHT**:

| Rank | Feature | Importance | Std |
|------|---------|------------|-----|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Physics interpretation**:

---

### Test 2.4: Feature Overlap Analysis

| Feature | Angle Rank | Depth Rank | L/R Rank | Shared? |
|---------|------------|------------|----------|---------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

**Findings**:
- Features shared across all targets:
- Features unique to angle:
- Features unique to depth:
- Features unique to left_right:

**Decision**: Use same features for all targets / Use target-specific features

---

## Phase 3: Model Comparison

Using the best feature set from Phase 1, compare different models.

### Test 3.1: LightGBM (Default Params)

**Command**:
```bash
uv run python -c "
from src.run_ablation import run_experiment
from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping
from src.data_loader import get_keypoint_columns

keypoint_cols = get_keypoint_columns()
init_keypoint_mapping(keypoint_cols)

result = run_experiment(
    name='lightgbm_default',
    extractor_func=extract_hybrid_features,
    smooth=False,  # or True based on Phase 1
    model_type='lightgbm',
    n_folds=5,
    cache_prefix='hybrid',
)
print(f'Scaled MSE: {result.scaled_mse:.6f}')
"
```

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Training time | |

---

### Test 3.2: XGBoost

**Command**:
```bash
uv run python src/run_ablation.py --experiments hybrid --model xgboost
```

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Training time | |

---

### Test 3.3: Model Comparison Summary

| Model | Scaled MSE | Training Time |
|-------|------------|---------------|
| LightGBM | | |
| XGBoost | | |

**Best model**:

---

## Phase 4: Hyperparameter Tuning

Using best feature set + best model from Phase 1-3.

### Test 4.1: Learning Rate Sweep

Test learning rates: 0.01, 0.02, 0.03, 0.05, 0.1

| Learning Rate | Scaled MSE |
|---------------|------------|
| 0.01 | |
| 0.02 | |
| 0.03 | |
| 0.05 | |
| 0.1 | |

**Best learning rate**:

---

### Test 4.2: Tree Depth / Leaves Sweep

For LightGBM, test num_leaves: 10, 15, 20, 31, 50

| num_leaves | Scaled MSE |
|------------|------------|
| 10 | |
| 15 | |
| 20 | |
| 31 | |
| 50 | |

**Best num_leaves**:

---

### Test 4.3: Regularization Sweep

Test reg_alpha and reg_lambda combinations

| reg_alpha | reg_lambda | Scaled MSE |
|-----------|------------|------------|
| 0 | 0 | |
| 0.1 | 0.1 | |
| 0.1 | 1.0 | |
| 1.0 | 1.0 | |

**Best regularization**:

---

### Test 4.4: Number of Estimators

| n_estimators | Scaled MSE | Training Time |
|--------------|------------|---------------|
| 200 | | |
| 500 | | |
| 1000 | | |
| 2000 | | |

**Best n_estimators**:

---

## Phase 5: Test Set Submission

Generate predictions on actual test set and submit to leaderboard.

### Test 5.1: Generate Submission

**Command**:
```bash
uv run python src/predict.py --model [best_model_path] --output output/submission_physics_v1.csv
```

Or use custom prediction script:
```bash
uv run python -c "
from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping
from src.data_loader import get_keypoint_columns, iterate_shots, load_metadata
from src.train_separate import SeparateTargetModels
import numpy as np
import pandas as pd
import lightgbm as lgb

# Initialize
keypoint_cols = get_keypoint_columns()
init_keypoint_mapping(keypoint_cols)

# Load and extract training features
print('Extracting training features...')
train_features = []
train_targets = []
train_meta = load_metadata(train=True)

for metadata, timeseries in iterate_shots(train=True, chunk_size=20):
    features = extract_hybrid_features(timeseries, metadata['participant_id'], smooth=False)
    train_features.append(features)
    train_targets.append([metadata['angle'], metadata['depth'], metadata['left_right']])

feature_names = sorted(train_features[0].keys())
X_train = np.array([[f.get(n, np.nan) for n in feature_names] for f in train_features], dtype=np.float32)
y_train = np.array(train_targets, dtype=np.float32)

# Impute NaN
for i in range(X_train.shape[1]):
    col = X_train[:, i]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        X_train[nan_mask, i] = np.nanmedian(col)

# Train final model
print('Training final model...')
model = SeparateTargetModels(
    lgb.LGBMRegressor,
    {'n_estimators': 500, 'num_leaves': 20, 'learning_rate': 0.02, 'verbose': -1}
)
model.fit(X_train, y_train, feature_names)

# Extract test features
print('Extracting test features...')
test_features = []
test_meta = load_metadata(train=False)

for metadata, timeseries in iterate_shots(train=False, chunk_size=20):
    features = extract_hybrid_features(timeseries, metadata['participant_id'], smooth=False)
    test_features.append(features)

X_test = np.array([[f.get(n, np.nan) for n in feature_names] for f in test_features], dtype=np.float32)

# Impute using training medians
for i in range(X_test.shape[1]):
    col = X_test[:, i]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        X_test[nan_mask, i] = np.nanmedian(X_train[:, i])

# Predict
print('Generating predictions...')
y_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': test_meta['id'],
    'angle': y_pred[:, 0],
    'depth': y_pred[:, 1],
    'left_right': y_pred[:, 2]
})
submission.to_csv('output/submission_hybrid_v1.csv', index=False)
print('Saved to output/submission_hybrid_v1.csv')
print(submission.head())
"
```

**Submission file**: output/submission_hybrid_v1.csv

---

### Test 5.2: Leaderboard Result

| Submission | CV Scaled MSE | Leaderboard MSE | Gap |
|------------|---------------|-----------------|-----|
| submission_hybrid_v1.csv | | | |

**Analysis**: CV vs Leaderboard difference indicates overfitting / underfitting / good generalization

---

## Phase 6: Error Analysis

Analyze where the model fails to guide next improvements.

### Test 6.1: Per-Participant Performance

| Participant | Angle MSE | Depth MSE | L/R MSE | Total MSE |
|-------------|-----------|-----------|---------|-----------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Worst participant**:
**Hypothesis for why**:

---

### Test 6.2: Per-Target Analysis

| Target | MSE | % of Total Error |
|--------|-----|------------------|
| Angle | | |
| Depth | | |
| Left/Right | | |

**Hardest target**:
**Hypothesis**:

---

### Test 6.3: Residual Analysis

For the worst predictions (top 10% error):
- Are they from specific participants?
- Are they from specific shot types?
- What features are unusual?

**Observations**:
1.
2.
3.

---

## Phase 7: Iteration Tracking

Track improvements across iterations.

| Version | Features | Model | Params | CV MSE | LB MSE | Notes |
|---------|----------|-------|--------|--------|--------|-------|
| v1 | Hybrid (132) | LightGBM | default | | | Initial physics approach |
| v2 | | | | | | |
| v3 | | | | | | |
| v4 | | | | | | |

---

## Phase 8: Frame Attention (Deep Learning)

Alternative approach: Learn which frames matter using temporal attention on raw time series.

**Prerequisites**: Complete Phases 1-6 first. Only proceed if:
- GBM approach plateaus
- CV MSE < 0.020 achieved
- Want to try ensemble with deep learning

### Test 8.1: Baseline LSTM

Simple LSTM on raw time series without attention.

**Architecture**:
```
Input: (batch, 240, 207)  # frames x features
    ↓
LSTM(hidden=128, layers=2, bidirectional=True)
    ↓
Last hidden state → FC(256) → FC(3)
    ↓
Output: (angle, depth, left_right)
```

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Training time | |
| Epochs to converge | |

---

### Test 8.2: LSTM with Frame Attention

Add attention to learn which frames matter.

**Architecture**:
```
Input: (batch, 240, 207)
    ↓
LSTM(hidden=128, layers=2, bidirectional=True)
    ↓
Outputs: (batch, 240, 256)  # hidden states for each frame
    ↓
Attention: learn weights for each frame
    ↓
Weighted sum of hidden states
    ↓
FC(256) → FC(3)
```

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Training time | |

**Attention visualization**: Which frames have highest attention weights?

| Frame Range | Avg Attention Weight | Interpretation |
|-------------|---------------------|----------------|
| 0-60 (prep) | | |
| 60-120 (load) | | |
| 120-180 (propulsion) | | |
| 180-240 (release) | | |

**Key finding**: Frames _____ to _____ matter most.

---

### Test 8.3: Per-Target Frame Attention

Train separate attention models for each target.

| Target | Peak Attention Frame | Interpretation |
|--------|---------------------|----------------|
| Angle | | |
| Depth | | |
| Left/Right | | |

**Finding**: Do different targets need different frames?

---

### Test 8.4: 1D CNN + Attention

Replace LSTM with 1D CNN for faster training.

**Architecture**:
```
Input: (batch, 240, 207)
    ↓
Conv1D(207→128, kernel=7) → BN → ReLU → MaxPool
Conv1D(128→256, kernel=5) → BN → ReLU → MaxPool
    ↓
Temporal attention over conv features
    ↓
FC(256) → FC(3)
```

| Metric | Result |
|--------|--------|
| Scaled MSE | |
| Training time | |

---

### Test 8.5: Ensemble GBM + Frame Attention

Combine best GBM model with best attention model.

| Model | Weight | Individual MSE |
|-------|--------|----------------|
| GBM (hybrid features) | | |
| Frame Attention | | |
| **Ensemble** | - | |

**Finding**: Does ensemble improve over best single model?

---

### Phase 8 Summary

| Model | Scaled MSE | Params | Training Time |
|-------|------------|--------|---------------|
| Baseline LSTM | | | |
| LSTM + Attention | | | |
| 1D CNN + Attention | | | |
| GBM (best) | | | |
| Ensemble | | | |

**Decision**: Use GBM only / Use DL only / Use ensemble

---

## Decision Framework

### After Full Factorial Experiment

```
If physics-only wins:
  -> Use ~40 features, simpler model
  -> Focus on feature refinement

If hybrid wins:
  -> Use ~130 features
  -> Consider feature selection to reduce further

If baseline wins:
  -> Physics hypothesis wrong
  -> Investigate why generic stats work
```

### Per-Target Decision Tree

For each target independently:

```
1. Find best (features, algorithm, training) from factorial
2. If multiple configs within 2*std of best:
   - Prefer simpler (fewer features)
   - Prefer faster (Ridge > GBM)
3. If left_right prefers shared but others prefer per-participant:
   - Use hybrid model architecture
```

### If CV MSE > 0.030:
- [ ] Review feature extraction for bugs
- [ ] Check for data leakage
- [ ] Verify release frame detection

### If CV MSE 0.020-0.030:
- [ ] Run full factorial if not done
- [ ] Hyperparameter tuning on best configs
- [ ] Try ensemble of top 3 configs

### If CV MSE < 0.020:
- [ ] Submit to leaderboard
- [ ] Fine-tune per-participant errors
- [ ] Try deep learning (Phase 8)

### If LB MSE >> CV MSE (overfitting):
- [ ] Increase regularization
- [ ] Prefer simpler features (physics over hybrid)
- [ ] Prefer shared over per-participant

### If LB MSE << CV MSE:
- [ ] Good - model generalizes well
- [ ] CV may be too pessimistic

---

## Commands Reference

**Run full factorial experiment** (all 40 feature x algorithm x training combinations):
```bash
uv run python src/run_factorial.py --all
```

**Run specific feature set with all algorithms and training approaches**:
```bash
uv run python src/run_factorial.py --features physics
uv run python src/run_factorial.py --features hybrid
uv run python src/run_factorial.py --features baseline
uv run python src/run_factorial.py --features stats_full
```

**Run legacy ablation experiments** (subset of factorial):
```bash
uv run python src/run_ablation.py --experiments all --model lightgbm
uv run python src/run_ablation.py --experiments physics --model lightgbm
uv run python src/run_ablation.py --experiments hybrid --model lightgbm
uv run python src/run_ablation.py --experiments baseline --model lightgbm
```

**Generate submission with per-target optimized config**:
```bash
uv run python src/predict.py --config output/best_config.json
```

**Generate submission with specific config**:
```bash
uv run python src/predict.py --features hybrid --algorithm lightgbm --training per-participant
```

**Clear feature cache** (if changing feature extraction):
```bash
rm output/features_*.pkl
```

**Run permutation importance for per-target feature analysis**:
```bash
uv run python -c "
from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping
from src.data_loader import get_keypoint_columns, iterate_shots, load_metadata
from sklearn.inspection import permutation_importance
import numpy as np
import lightgbm as lgb

# Initialize
keypoint_cols = get_keypoint_columns()
init_keypoint_mapping(keypoint_cols)

# Load data
print('Loading data...')
meta = load_metadata(train=True)
all_features = []
all_targets = []

for metadata, timeseries in iterate_shots(train=True, chunk_size=20):
    features = extract_hybrid_features(timeseries, metadata['participant_id'], smooth=False)
    all_features.append(features)
    all_targets.append([metadata['angle'], metadata['depth'], metadata['left_right']])

feature_names = sorted(all_features[0].keys())
X = np.array([[f.get(n, np.nan) for n in feature_names] for f in all_features], dtype=np.float32)
y = np.array(all_targets, dtype=np.float32)

# Impute NaN
for i in range(X.shape[1]):
    col = X[:, i]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        X[nan_mask, i] = np.nanmedian(col)

# Train models and compute permutation importance for each target
targets = ['angle', 'depth', 'left_right']

for t_idx, target in enumerate(targets):
    print(f'\n=== {target.upper()} ===')

    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=20, learning_rate=0.03, verbose=-1)
    model.fit(X, y[:, t_idx])

    # Permutation importance
    result = permutation_importance(model, X, y[:, t_idx], n_repeats=10, random_state=42, n_jobs=-1)

    # Sort by importance
    sorted_idx = result.importances_mean.argsort()[::-1]

    print(f'Top 10 features for {target}:')
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f'  {rank}. {feature_names[idx]}: {result.importances_mean[idx]:.4f} +/- {result.importances_std[idx]:.4f}')
"
```

---

## Stage 2: Optimization Tests (After Foundation is Complete)

**Prerequisites**: Complete Phases 0-7 first. Only proceed when:
- Confident in best algorithm choice
- Confident in core feature approach
- CV MSE is stable and understood
- Per-target analysis complete

---

### Stage 2.1: Participant-Specific Analysis

**Goal**: Determine if different participants need different models/features.

#### Test: Per-Participant Model Performance

```python
# Train on all participants, evaluate per-participant
for participant in [1, 2, 3, 4, 5]:
    mask = participant_ids == participant
    participant_mse = mean_squared_error(y_true[mask], y_pred[mask])
```

| Participant | angle MSE | depth MSE | left_right MSE | Total |
|-------------|-----------|-----------|----------------|-------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

#### Test: Participant-Specific Feature Importance

Do the same features matter for each participant?

| Feature | P1 Rank | P2 Rank | P3 Rank | P4 Rank | P5 Rank | Variance |
|---------|---------|---------|---------|---------|---------|----------|
| | | | | | | |

#### Test: Participant-Normalized Features

```python
# Z-score features within each participant
for p in participants:
    mask = participant_ids == p
    X[mask] = (X[mask] - X[mask].mean(0)) / (X[mask].std(0) + 1e-8)
```

| Approach | CV MSE | vs Baseline |
|----------|--------|-------------|
| Raw features | | |
| Participant-normalized | | |

---

### Stage 2.2: Ensemble Experiments

**Goal**: Determine if combining models improves over single best model.

#### Test: Simple Averaging

```python
pred_final = (pred_lgbm + pred_xgb + pred_catboost) / 3
```

| Ensemble | CV MSE | vs Best Single |
|----------|--------|----------------|
| LightGBM only | | baseline |
| LightGBM + XGBoost | | |
| LightGBM + CatBoost | | |
| All three averaged | | |

#### Test: Weighted Averaging

```python
# Weights inversely proportional to CV error
weights = 1 / np.array([lgbm_mse, xgb_mse, cat_mse])
weights = weights / weights.sum()
pred_final = weights[0]*pred_lgbm + weights[1]*pred_xgb + weights[2]*pred_catboost
```

| Weighting | CV MSE |
|-----------|--------|
| Equal (1/3 each) | |
| Inverse MSE weighted | |
| Optimized weights | |

#### Test: Stacking Ensemble

```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('lgbm', LGBMRegressor(**params)),
    ('xgb', XGBRegressor(**params)),
    ('cat', CatBoostRegressor(**params)),
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
```

| Meta-Learner | CV MSE |
|--------------|--------|
| Ridge | |
| LightGBM (shallow) | |
| Linear (no regularization) | |

#### Test: Per-Target Ensembles

Different ensemble for each target:

| Target | Best Ensemble | CV MSE |
|--------|---------------|--------|
| angle | | |
| depth | | |
| left_right | | |

---

### Stage 2.3: Advanced Feature Sets

**Goal**: Test if sophisticated features improve over simple baseline.

#### Test: tsfresh Automated Features

```python
from tsfresh import extract_features
from tsfresh.feature_selection import select_features

# Extract all 794 features
features = extract_features(df, column_id='shot_id', column_sort='frame')

# Filter to relevant
selected = select_features(features, y, fdr_level=0.05)
```

| Feature Set | # Features | CV MSE |
|-------------|------------|--------|
| Baseline (mean+last) | 414 | |
| tsfresh (all) | 794 | |
| tsfresh (selected) | ? | |
| Baseline + tsfresh selected | ? | |

#### Test: Wavelet Features

```python
import pywt

for joint in key_joints:
    coeffs = pywt.wavedec(joint_trajectory, 'sym4', level=3)
    # Extract energy, entropy per level
```

| Feature Set | CV MSE |
|-------------|--------|
| Baseline | |
| Baseline + Wavelet | |
| Wavelet only | |

#### Test: FFT/Frequency Features

```python
fft_coeffs = np.fft.rfft(trajectory)
power_spectrum = np.abs(fft_coeffs)**2
# Dominant frequency, spectral centroid, band powers
```

| Feature Set | CV MSE |
|-------------|--------|
| Baseline | |
| Baseline + FFT | |

#### Test: Physics-Informed Features

```python
# Estimated launch angle, velocity, projectile motion parameters
est_launch_angle = np.arctan(vz / np.sqrt(vx**2 + vy**2))
est_launch_velocity = np.sqrt(vx**2 + vy**2 + vz**2)
```

| Feature Set | CV MSE |
|-------------|--------|
| Baseline | |
| Baseline + Physics | |
| Physics only | |

---

### Stage 2.4: Regressor Chain (Target Correlation)

**Goal**: Exploit correlations between targets.

```python
from sklearn.multioutput import RegressorChain

# Test different orderings
orders = [
    [0, 1, 2],  # angle -> depth -> left_right
    [0, 2, 1],  # angle -> left_right -> depth
    [1, 0, 2],  # depth -> angle -> left_right
    [2, 0, 1],  # left_right -> angle -> depth
]
```

| Chain Order | CV MSE | vs Separate Models |
|-------------|--------|-------------------|
| Separate (no chain) | | baseline |
| angle -> depth -> l/r | | |
| angle -> l/r -> depth | | |
| depth -> angle -> l/r | | |
| l/r -> angle -> depth | | |

---

### Stage 2.5: Final Hyperparameter Tuning

**Goal**: Squeeze last bits of performance from best configuration.

Only do this AFTER all other decisions are made.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.02, 0.05],
    'num_leaves': [15, 31, 50],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
}

grid = GridSearchCV(model, param_grid, cv=logo, scoring='neg_mean_squared_error')
```

| Target | Best Params | CV MSE |
|--------|-------------|--------|
| angle | | |
| depth | | |
| left_right | | |

---

## Stage 2 Summary

| Optimization | Attempted | Result | Keep? |
|--------------|-----------|--------|-------|
| Participant normalization | | | |
| Simple ensemble | | | |
| Stacking ensemble | | | |
| tsfresh features | | | |
| Wavelet features | | | |
| FFT features | | | |
| Physics features | | | |
| Regressor chain | | | |
| Hyperparameter tuning | | | |

**Final Configuration**:
- Algorithm:
- Features:
- Ensemble:
- Per-target customization:

**Final CV MSE**:
**Final LB MSE**:
