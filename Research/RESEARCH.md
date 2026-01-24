# SPLxUTSPAN 2026 Data Challenge - Comprehensive Research Collection

## Executive Summary

This document compiles all proven insights, algorithm breakthroughs, and innovative techniques for predicting basketball free throw outcomes (angle, depth, left_right) from motion capture data.

**Your Data:**
- 344 training shots, 112 test shots, 5 participants
- 69 body parts x 3 coords = 207 features per frame
- 240 frames at 60 FPS (4 seconds)
- Targets: angle (entry angle), depth (short/long), left_right (lateral deviation)

**Current State:** ~0.035 MSE | **Target:** <0.020 MSE | **Leader:** 0.008381 MSE

---

## 1. Competition-Specific Data Analysis

### Dataset Structure
- **Training**: 344 shots across 5 participants (66-74 shots each)
- **Test**: 112 shots
- **Features**: 69 body parts x 3 coordinates (x, y, z) = 207 time series per shot
- **Time Series Length**: 240 frames per shot (uniform)
- **Frame Rate**: 60 FPS (4 seconds total per shot)
- **Targets**:
  - `angle`: Shot entry angle (23.78-56.47 degrees, mean 45.48)
  - `depth`: Distance from center - short/long (-10.17 to 24.97 inches, mean 9.66)
  - `left_right`: Horizontal deviation (-12.98 to 10.15 inches, mean -0.78)

### Target Scaling (MinMaxScaler, range 0-1)
- **Angle**: Original [30, 60] degrees -> [0, 1]
- **Depth**: Original [-12, 30] inches -> [0, 1]
- **Left/Right**: Original [-16, 16] inches -> [0, 1]

### Participant Statistics
| Participant | Shots | Angle (mean/std) | Depth (mean/std) | L/R (mean/std) |
|-------------|-------|------------------|------------------|----------------|
| 1 | 70 | 44.38/1.30 | 11.33/4.55 | -0.84/4.13 |
| 2 | 66 | 42.62/2.07 | 10.93/4.26 | -0.10/3.74 |
| 3 | 68 | 48.02/1.64 | 9.54/2.32 | -1.25/2.85 |
| 4 | 67 | 52.25/2.70 | 9.09/4.85 | -0.64/3.93 |
| 5 | 74 | 40.63/4.10 | 7.57/8.16 | -1.04/4.16 |

**Key Insight**: Each participant has distinct shooting characteristics. Participant-aware modeling is critical.

### Body Part Keypoints (69 total)
**Core body (23)**: nose, left/right_eye, left/right_ear, left/right_shoulder, left/right_elbow, left/right_wrist, left/right_hip, left/right_knee, left/right_ankle, left/right_big_toe, left/right_small_toe, left/right_heel, mid_hip, neck

**Left hand (22)**: left_wrist_2, left_first_finger (cmc/mcp/ip/distal), left_second_finger (mcp/pip/dip/distal), left_third_finger, left_fourth_finger, left_fifth_finger, left_thumb, left_pinky

**Right hand (22)**: right_wrist_2, right_first_finger (cmc/mcp/ip/distal), right_second_finger (mcp/pip/dip/distal), right_third_finger, right_fourth_finger, right_fifth_finger, right_thumb, right_pinky

---

## 2. CRITICAL RESEARCH FINDINGS: Basketball Free Throw Biomechanics

### 2.1 The Two Most Predictive Variables (PROVEN)

**Research Source:** [Key Kinematic Components for Optimal Free Throw](https://www.semanticscholar.org/paper/Key-Kinematic-Components-for-Optimal-Basketball-Cabarkapa-Fry/5e189104dad27f032ec0bf15c99788638b68afae)

> "Two dependent variables that demonstrated the greatest impact in predicting between proficient and non-proficient free throw shooters were **elbow flexion** and **forearm angle**. Proficient shooters had greater elbow flexion and less lateral elbow deviation from the imaginary vertical axis when compared to non-proficient shooters."

**Implementation:**
```python
# Elbow flexion angle
v1 = elbow_pos - shoulder_pos
v2 = wrist_pos - elbow_pos
elbow_angle = arccos(dot(v1, v2) / (|v1| * |v2|))

# Elbow lateral deviation (from vertical)
elbow_lateral_dev = distance from elbow to vertical plane through shoulder
```

### 2.2 Release Velocity Standard Deviation - HIGHEST CORRELATION

**Research Source:** [Release Variability and Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/)

> "Release velocity standard deviation alone accurately predicts shooting performance."
> - r = -0.96 for three-point shots
> - r = -0.88 for free throws
> "As a participant's velocity SD decreases, their shooting accuracy increases."

**Implementation:**
```python
velocity_mag = sqrt(vx**2 + vy**2 + vz**2)
velocity_sd = np.std(velocity_mag)  # HIGHLY PREDICTIVE
```

### 2.3 Proficient vs Non-Proficient Shooter Differences

**Research Source:** [Proficient Free-Throw Shooters Analysis](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2023.1208915/full)

Key differentiators:
1. **Release height**: Proficient shooters have GREATER release height (d = 0.438)
2. **Trunk lean**: Proficient shooters have LESS forward trunk lean at release (d = 0.880)
3. **Control**: Lower knee and center of mass angular velocities (more controlled motion)

**Implementation:**
```python
release_height = wrist_z[release_frame]
trunk_lean = arccos(dot(neck - mid_hip, vertical_axis))
```

### 2.4 Optimal Release Parameters (Physics)

**Research Sources:** Multiple biomechanics studies

- **Optimal release angle**: 51-60 degrees (closer = higher angle, farther = lower angle)
- **Optimal release velocity**: ~7.3 m/s
- **Optimal backspin**: ~3 Hz

**Key insight:** The relationship is:
- **entry_angle** ~ f(release_angle, arc_height)
- **depth** ~ f(release_velocity, release_angle, release_height)
- **left_right** ~ f(elbow_alignment, horizontal_velocity_x)

### 2.5 Motor Control and Variability

**Research Source:** [Motor Control and Shooting Consistency](https://www.biorxiv.org/content/10.1101/793513v1.full)

> "The role of sensorimotor noise in redundant tasks and its proportionality with force and velocity have been widely recognized. The distal high variability is a determinant of free-throw effectiveness."

Key finding: The most successful shooters show MORE variability in triceps brachii activation timing - they adapt dynamically rather than being rigid.

---

## 3. Physics-Based Feature Engineering (HIGH IMPACT)

### 3.1 Release Point Detection - CRITICAL

The release point is the most predictive moment. Test ALL methods:

| Method | Description | Code |
|--------|-------------|------|
| Max wrist_z | Highest wrist position | `np.argmax(wrist_z)` |
| Max velocity | Peak wrist velocity | `np.argmax(velocity_mag)` |
| Velocity direction change | Vertical velocity peaks | `np.argmax(velocity_z)` |
| Acceleration zero-crossing | Upward accel stops | `np.where(np.diff(np.sign(accel_z)))[0]` |
| Elbow extension peak | Max extension rate | `np.argmax(elbow_angular_velocity)` |

### 3.2 Features at Release Point

```python
# MUST-HAVE release features
features = {
    # Position at release
    'right_wrist_x_release': right_wrist_x[release_frame],
    'right_wrist_y_release': right_wrist_y[release_frame],
    'right_wrist_z_release': right_wrist_z[release_frame],
    'release_height': right_wrist_z[release_frame],

    # Velocity at release
    'right_wrist_vz_release': velocity_z[release_frame],
    'right_wrist_v_mag_release': velocity_mag[release_frame],

    # HIGHLY PREDICTIVE: Velocity consistency
    'right_wrist_v_sd': np.std(velocity_mag),
    'right_wrist_vz_sd': np.std(velocity_z),

    # Joint angles at release
    'elbow_angle_release': compute_elbow_angle(release_frame),
    'shoulder_angle_release': compute_shoulder_angle(release_frame),
    'trunk_lean_release': compute_trunk_lean(release_frame),
    'elbow_lateral_dev': compute_elbow_lateral_deviation(release_frame),

    # Estimated launch parameters
    'est_launch_angle': np.arctan(vz / np.sqrt(vx**2 + vy**2)),
    'est_launch_velocity': np.sqrt(vx**2 + vy**2 + vz**2),
}
```

### 3.3 Velocity Features (First Derivative)

```python
# Central difference for smoother velocity
velocity = np.zeros_like(position)
velocity[1:-1] = (position[2:] - position[:-2]) / (2 * dt)
velocity[0] = position[1] - position[0]
velocity[-1] = position[-1] - position[-2]

# Features for each keypoint
features = {
    'mean_velocity': np.mean(velocity),
    'max_velocity': np.max(np.abs(velocity)),
    'velocity_at_release': velocity[release_frame],
    'velocity_sd': np.std(velocity),  # CRITICAL
    'time_to_max_velocity': np.argmax(velocity),
}
```

### 3.4 Acceleration Features (Second Derivative)

```python
acceleration = np.diff(velocity, axis=0) / dt

features = {
    'mean_acceleration': np.mean(acceleration),
    'max_acceleration': np.max(np.abs(acceleration)),
    'acceleration_at_release': acceleration[release_frame],
    'jerk_mean': np.mean(np.diff(acceleration)),  # smoothness
}
```

---

## 4. Biomechanical Joint Angle Features

### 4.1 Critical Angles (from research)

**1. Elbow Flexion Angle** (MOST PREDICTIVE)
```python
def compute_elbow_angle(frame, shoulder, elbow, wrist):
    v1 = elbow[frame] - shoulder[frame]
    v2 = wrist[frame] - elbow[frame]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
```

**2. Shoulder Flexion Angle** (arm elevation)
```python
def compute_shoulder_angle(frame, mid_hip, shoulder, elbow):
    v1 = shoulder[frame] - mid_hip[frame]  # trunk
    v2 = elbow[frame] - shoulder[frame]    # upper arm
    # Same angle calculation
```

**3. Knee Flexion Angle**
```python
def compute_knee_angle(frame, hip, knee, ankle):
    v1 = knee[frame] - hip[frame]
    v2 = ankle[frame] - knee[frame]
    # Same angle calculation
```

**4. Trunk Lean Angle** (KEY DIFFERENTIATOR)
```python
def compute_trunk_lean(frame, mid_hip, neck):
    trunk_vector = neck[frame] - mid_hip[frame]
    vertical = np.array([0, 0, 1])
    cos_angle = np.dot(trunk_vector, vertical) / np.linalg.norm(trunk_vector)
    return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
```

**5. Elbow Lateral Deviation** (from vertical axis)
```python
def compute_elbow_lateral_deviation(frame, shoulder, elbow):
    # Distance from elbow to vertical line through shoulder
    shoulder_xy = shoulder[frame][:2]
    elbow_xy = elbow[frame][:2]
    return np.linalg.norm(elbow_xy - shoulder_xy)
```

### 4.2 For Each Angle, Extract:
- Angle at release
- Min/max angle during shot
- Range of motion (max - min)
- Angular velocity at release
- Time of max flexion/extension

---

## 5. Temporal Phase Features

### 5.1 Fixed Phase Segmentation
| Phase | Frames | Description |
|-------|--------|-------------|
| Preparation | 0-60 | Stance, ball positioning |
| Loading | 60-120 | Knee bend, arm preparation, energy storage |
| Propulsion | 120-180 | Leg drive, arm extension, energy transfer |
| Release | 180-240 | Ball release, follow-through |

### 5.2 Dynamic Phase Detection (Better)
```python
def detect_phases(knee_angle, com_velocity, wrist_velocity):
    # Loading start: Knee flexion begins
    loading_start = np.where(np.diff(knee_angle) < -threshold)[0][0]

    # Propulsion start: CoM starts moving up
    propulsion_start = np.where(com_velocity > 0)[0][0]

    # Release: Max wrist velocity
    release = np.argmax(wrist_velocity)

    return loading_start, propulsion_start, release
```

### 5.3 Phase-Based Features
For each phase, extract:
- Mean/SD/min/max of key joint positions
- Mean/SD of velocities
- Phase duration
- Phase transition smoothness

---

## 6. Advanced Signal Processing Features

### 6.1 Wavelet Transform Features (HIGH POTENTIAL)

**Research Source:** [Wavelet Features for Motion](https://www.nature.com/articles/s41598-025-22701-z)

> "Multi-resolution wavelet decomposition captures both fine and coarse motion features."

```python
import pywt

def extract_wavelet_features(signal, wavelet='sym4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    for i, c in enumerate(coeffs):
        features[f'wavelet_energy_level_{i}'] = np.sum(c**2)
        features[f'wavelet_entropy_level_{i}'] = -np.sum((c**2) * np.log(c**2 + 1e-10))
        features[f'wavelet_std_level_{i}'] = np.std(c)
        features[f'wavelet_kurtosis_level_{i}'] = kurtosis(c)
    return features
```

Apply to: wrist_z, elbow_z, shoulder_z, velocity profiles

### 6.2 Fourier Transform Features (FFT)

```python
def extract_fft_features(signal):
    fft_coeffs = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_coeffs)**2
    freqs = np.fft.rfftfreq(len(signal), d=1/60)  # 60 FPS

    features = {
        'dominant_freq': freqs[np.argmax(power_spectrum)],
        'power_0_2hz': np.sum(power_spectrum[(freqs >= 0) & (freqs <= 2)]),
        'power_2_5hz': np.sum(power_spectrum[(freqs > 2) & (freqs <= 5)]),
        'power_5_10hz': np.sum(power_spectrum[(freqs > 5) & (freqs <= 10)]),
        'spectral_centroid': np.sum(freqs * power_spectrum) / np.sum(power_spectrum),
    }
    return features
```

### 6.3 tsfresh Automated Feature Extraction

**Resource:** [tsfresh Documentation](https://tsfresh.readthedocs.io)

> "Automatically extracts 794 time series features with built-in relevance filtering."

```python
from tsfresh import extract_features
from tsfresh.feature_selection import select_features

# Format data for tsfresh
df = pd.DataFrame({
    'shot_id': shot_ids,
    'frame': frames,
    'right_wrist_z': values,
    # ... other columns
})

# Extract ALL features
features = extract_features(df, column_id='shot_id', column_sort='frame')

# Filter to relevant features
selected = select_features(features, y, fdr_level=0.05)
```

---

## 7. Kinematic Chain Coordination Features

### 7.1 Proximal-to-Distal Sequencing

**Key Insight:** Skilled shooters follow ankle -> knee -> hip -> shoulder -> elbow -> wrist timing pattern.

```python
def extract_timing_features(joints_velocities):
    # Find time of max velocity for each joint
    times = {}
    for joint_name, velocity in joints_velocities.items():
        times[joint_name] = np.argmax(velocity)

    # Time lags (should be positive for proper sequencing)
    features = {
        'hip_to_shoulder_lag': times['shoulder'] - times['hip'],
        'shoulder_to_elbow_lag': times['elbow'] - times['shoulder'],
        'elbow_to_wrist_lag': times['wrist'] - times['elbow'],
        'total_sequencing_time': times['wrist'] - times['hip'],
    }
    return features
```

### 7.2 Joint Coordination Cross-Correlation

```python
def cross_correlation_features(joint1_velocity, joint2_velocity):
    correlation = np.correlate(joint1_velocity, joint2_velocity, mode='full')
    lag = np.argmax(correlation) - len(joint1_velocity) + 1
    max_corr = np.max(correlation)
    return {'lag': lag, 'max_correlation': max_corr}
```

---

## 8. Algorithm Recommendations

### 8.1 Why Gradient Boosting Beats Deep Learning for Your Data

**Research Source:** [Gradient Boosting vs Deep Learning](https://forecastegy.com/posts/gradient-boosting-vs-deep-learning-tabular-data/)

> "Deep learning can struggle on small datasets. Tree-based models can effectively ignore uninformative features by splitting on the most relevant ones."

With 344 samples:
- **CatBoost**: Best default performance, handles small datasets well
- **LightGBM**: Fast training, good for iteration
- **XGBoost**: Robust, well-documented

### 8.2 Recommended Hyperparameters

**CatBoost:**
```python
{
    'iterations': 500-1000,
    'learning_rate': 0.02-0.05,
    'depth': 4-6,
    'l2_leaf_reg': 3-10,
    'random_strength': 1,
    'verbose': False,
}
```

**LightGBM:**
```python
{
    'n_estimators': 500-1000,
    'learning_rate': 0.02-0.05,
    'num_leaves': 15-31,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 10-20,
    'verbose': -1,
}
```

### 8.3 Multi-Target Regression Strategies

**Option A: Separate Models (RECOMMENDED START)**
```python
# Train 3 independent models
models = {
    'angle': LGBMRegressor(**params),
    'depth': LGBMRegressor(**params),
    'left_right': LGBMRegressor(**params),
}
for target, model in models.items():
    model.fit(X, y[target])
```

**Option B: Regressor Chain (EXPLOIT TARGET CORRELATIONS)**

**Research Source:** [Multi-Target Regression Chains](https://link.springer.com/article/10.1007/s10994-018-5744-y)

> "Correlation Regressor Chains improve prediction performance by exploiting correlations among targets."

```python
from sklearn.multioutput import RegressorChain

# Try different orderings
chain = RegressorChain(LGBMRegressor(**params), order=[0, 2, 1])
chain.fit(X, y)  # angle -> left_right -> depth
```

**Option C: Ensemble of Regressor Chains (ERC)**
```python
# Multiple chains with different orderings
orders = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [1, 2, 0], [2, 1, 0]]
chains = [RegressorChain(LGBMRegressor(**params), order=o) for o in orders]
# Average predictions
```

### 8.4 Stacking Ensemble

**Research Source:** [Stacking Ensemble Methods](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

```
Level 0 (Base Models):
- CatBoost with physics features
- LightGBM with physics features
- XGBoost with all features
- Ridge with physics features

Level 1 (Meta-Learner):
- Ridge Regression (prevents overfitting)
```

```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('catboost', CatBoostRegressor(**cat_params)),
    ('lgbm', LGBMRegressor(**lgb_params)),
    ('xgb', XGBRegressor(**xgb_params)),
    ('ridge', Ridge(alpha=1.0)),
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
)
```

---

## 9. Innovative/Creative Techniques

### 9.1 Physics-Informed Neural Networks (PINN)

**Research Source:** [Physics-Informed Neural Networks](https://arxiv.org/abs/2506.12029)

> "PINN reduces average displacement errors by up to 32%."

```python
# Physics loss: enforce projectile motion constraints
def physics_loss(pred, release_params):
    # depth should follow projectile motion
    expected_depth = projectile_model(
        release_params['velocity'],
        release_params['angle'],
        release_params['height']
    )
    return mse(pred['depth'], expected_depth)

total_loss = data_loss + lambda_physics * physics_loss
```

### 9.2 Symbolic Regression for Feature Discovery

**Research Source:** [PySR for Equation Discovery](https://arxiv.org/abs/2508.20257)

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "sqrt", "square"],
    populations=15,
)
model.fit(physics_features, target)
print(model.equations_)  # Discover interpretable formulas
```

### 9.3 Dynamic Time Warping (DTW) Similarity

**Concept:** Compare shots to "best shot" templates.

```python
from dtaidistance import dtw

# Find best shots (lowest error in training)
best_shot_templates = select_best_shots(X_train, y_train)

# For each shot, compute DTW distance to best shots
def dtw_features(shot_trajectory, templates):
    features = {}
    for i, template in enumerate(templates):
        features[f'dtw_dist_to_best_{i}'] = dtw.distance(shot_trajectory, template)
    features['dtw_min'] = min(features.values())
    return features
```

### 9.4 Graph Neural Networks for Skeleton

**Research Source:** [ST-GCN for Skeleton Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8952863/)

```python
# Build adjacency matrix based on skeleton structure
adjacency = build_skeleton_adjacency()  # 69x69 matrix

# ST-GCN layer
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, kernel_size=9)

    def forward(self, x):
        # x: (batch, frames, joints, channels)
        x = self.gcn(x)  # Spatial
        x = self.tcn(x)  # Temporal
        return x
```

### 9.5 Autoencoder Feature Extraction

**Research Source:** [LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)

```python
# LSTM Autoencoder
encoder = nn.Sequential(
    nn.LSTM(207, 128, batch_first=True),
    nn.LSTM(128, 64, batch_first=True),
    nn.LSTM(64, 32, batch_first=True),  # Bottleneck
)
decoder = nn.Sequential(
    nn.LSTM(32, 64, batch_first=True),
    nn.LSTM(64, 128, batch_first=True),
    nn.LSTM(128, 207, batch_first=True),
)

# Train to reconstruct, use 32-dim bottleneck as features
```

### 9.6 Residual Learning

```python
# Step 1: Physics baseline
physics_pred = physics_model(release_params)

# Step 2: ML predicts residual
residual = actual - physics_pred
ml_model.fit(features, residual)

# Step 3: Final = physics + ML residual
final_pred = physics_pred + ml_model.predict(features)
```

### 9.7 Participant-Specific Models

**Approach 1: Participant Embedding**
```python
# One-hot or learned embedding for participant
X_with_participant = np.hstack([X, participant_onehot])
```

**Approach 2: Feature Normalization by Participant**
```python
# Z-score within each participant
for p in participants:
    mask = participant_ids == p
    X[mask] = (X[mask] - X[mask].mean(0)) / (X[mask].std(0) + 1e-8)
```

---

## 10. Data Augmentation Techniques

**Research Source:** [Data Augmentation for Biomechanics](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327038)

> "Data augmentation improved accuracy up to 23% in motion capture prediction tasks."

### 10.1 Time Warping
```python
def time_warp(signal, warp_factor):
    original_indices = np.arange(len(signal))
    warped_indices = original_indices * warp_factor
    warped = np.interp(original_indices, warped_indices, signal)
    return warped
```

### 10.2 Noise Injection
```python
augmented = original + np.random.normal(0, noise_std, original.shape)
```

### 10.3 Mixup
```python
lambda_ = np.random.beta(0.2, 0.2)
mixed_X = lambda_ * X_a + (1 - lambda_) * X_b
mixed_y = lambda_ * y_a + (1 - lambda_) * y_b
```

### 10.4 Small Rotation
```python
# Rotate around vertical axis (small angle)
theta = np.random.uniform(-5, 5) * np.pi / 180
rotation_matrix = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
xy_rotated = xy @ rotation_matrix
```

---

## 11. Validation Strategy

### 11.1 Leave-One-Participant-Out (LOPO)

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
scores = []
for train_idx, val_idx in logo.split(X, y, groups=participant_ids):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    scores.append(compute_mse(pred, y_val))

print(f"Mean CV MSE: {np.mean(scores):.6f} +/- {np.std(scores):.6f}")
```

### 11.2 Nested CV for Hyperparameter Tuning

```python
# Outer loop: LOPO for unbiased estimate
# Inner loop: 4-fold for hyperparameter selection
from sklearn.model_selection import cross_val_score, GridSearchCV

for train_idx, test_idx in logo.split(X, y, groups=participant_ids):
    # Inner CV on training set only
    inner_cv = GroupKFold(n_splits=4)
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X[train_idx], y[train_idx], groups=participant_ids[train_idx])

    # Evaluate on test fold
    best_model = grid_search.best_estimator_
    score = best_model.score(X[test_idx], y[test_idx])
```

---

## 12. Limiting Factors & Potential Problems

### 12.1 Small Dataset Size (344 samples)
**Problem:** Insufficient for complex models
**Mitigations:**
- Simple models (shallow trees, high regularization)
- Feature selection to reduce dimensionality
- Ensemble multiple weak learners
- Data augmentation

### 12.2 Participant Variability (5 distinct styles)
**Problem:** Model might overfit to training participants
**Mitigations:**
- Participant-relative feature normalization
- LOPO validation
- Physics-based features (more generalizable)

### 12.3 Feature Multicollinearity
**Problem:** Many features are correlated
**Mitigations:**
- RFE (Recursive Feature Elimination)
- PCA on correlated groups
- L1/L2 regularization

### 12.4 Motion Capture Noise
**Problem:** Marker jitter, occlusions
**Mitigations:**
- Savitzky-Golay smoothing (window=5-11, polyorder=2-3)
- Compute derivatives after smoothing
- Use robust statistics (median)

### 12.5 Target Scale Differences
**Problem:** Different units/ranges
**Mitigations:**
- Train separate models per target
- Scale targets during training
- Use scaled MSE for comparison

---

## 13. Feature Priority Matrix

| Feature Category | Difficulty | Expected Impact | Priority |
|-----------------|-----------|----------------|---------|
| Release point detection | Low | HIGH | P0 |
| Elbow/shoulder angles at release | Low | HIGH | P0 |
| Velocity at release | Low | HIGH | P0 |
| Velocity SD (consistency) | Low | HIGH | P0 |
| Trunk lean angle | Low | MEDIUM-HIGH | P0 |
| Phase-based features | Medium | MEDIUM-HIGH | P1 |
| Kinematic chain timing | Medium | MEDIUM | P1 |
| Wavelet features | Medium | MEDIUM | P1 |
| tsfresh automated | Low | UNKNOWN | P1 |
| CatBoost tuning | Low | MEDIUM | P1 |
| Stacking ensemble | Medium | MEDIUM | P2 |
| RegressorChain | Low | MEDIUM | P2 |
| Participant normalization | Low | MEDIUM | P2 |
| GNN/Transformer | High | UNKNOWN | P3 |
| PINN | High | UNKNOWN | P3 |
| Symbolic regression | Medium | UNKNOWN | P3 |

---

## 14. Target-Specific Feature Hypotheses

### For ANGLE (entry angle):
- Release angle (primary)
- Arc height features
- Wrist snap velocity (rotation)
- Shoulder-to-wrist alignment

### For DEPTH (short/long):
- Release velocity magnitude (primary)
- Release angle
- Leg drive (CoM vertical velocity)
- Knee angle (power generation)

### For LEFT_RIGHT (lateral deviation):
- Elbow lateral alignment (primary)
- Shoulder rotation at release
- Wrist lateral position
- Guide hand position

---

## 15. Implementation Plan

### Phase 1: Quick Wins (1-2 days)
1. Implement release point detection (5 methods)
2. Add release-point features for all joints
3. Add elbow/shoulder/trunk angles at release
4. Test CatBoost with physics features
5. Compute velocity SD features

### Phase 2: Feature Expansion (2-3 days)
6. Add kinematic chain timing features
7. Add wavelet features for key joints
8. Try tsfresh automated extraction
9. Implement phase-based features
10. Add participant normalization

### Phase 3: Model Optimization (1-2 days)
11. Hyperparameter tuning with nested CV
12. Try RegressorChain with different orderings
13. Implement 2-level stacking ensemble
14. Per-target feature selection

### Phase 4: Innovation (1-2 days)
15. Try PINN for physics-informed learning
16. Try symbolic regression for feature discovery
17. Train LSTM/Transformer for ensemble diversity
18. Implement DTW similarity features

---

## 16. Sources

### Basketball Biomechanics
- [Key Kinematic Components for Optimal Free Throw](https://www.semanticscholar.org/paper/Key-Kinematic-Components-for-Optimal-Basketball-Cabarkapa-Fry/5e189104dad27f032ec0bf15c99788638b68afae)
- [Proficient Free-Throw Shooters Analysis](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2023.1208915/full)
- [Release Variability and Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/)
- [Motor Control and Shooting Consistency](https://www.biorxiv.org/content/10.1101/793513v1.full)
- [Kinematics of Arm Joint Motions](https://www.sciencedirect.com/science/article/pii/S187770581501471X)

### Machine Learning & Time Series
- [Gradient Boosting vs Deep Learning](https://forecastegy.com/posts/gradient-boosting-vs-deep-learning-tabular-data/)
- [tsfresh Documentation](https://tsfresh.readthedocs.io)
- [Wavelet Features for Motion](https://www.nature.com/articles/s41598-025-22701-z)
- [Multi-Target Regression Chains](https://link.springer.com/article/10.1007/s10994-018-5744-y)
- [Feature Ranking for Multi-target Regression](https://link.springer.com/chapter/10.1007/978-3-319-67786-6_13)

### Deep Learning & Advanced
- [Physics-Informed Neural Networks](https://arxiv.org/abs/2506.12029)
- [ST-GCN for Skeleton Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8952863/)
- [Data Augmentation for Biomechanics](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327038)
- [Stacking Ensemble Methods](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
- [LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)
- [Temporal Attention Mechanisms](https://arxiv.org/abs/2308.12874)

### Validation & Methods
- [Leave-One-Subject-Out CV](https://scikit-learn.org/stable/modules/cross_validation.html)
- [RFE Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Dynamic Time Warping](https://link.springer.com/article/10.1007/s11045-018-0611-3)

---

## 17. Verification Checklist

After implementing:
1. [ ] No NaN in features, reasonable value ranges
2. [ ] CV MSE improved vs baseline (0.035)
3. [ ] Top features include release-point and physics features
4. [ ] Error similar across all 5 participants
5. [ ] Per-target error analysis complete
6. [ ] Leaderboard submission, compare CV to LB

**Success Criteria:** CV scaled MSE < 0.020
