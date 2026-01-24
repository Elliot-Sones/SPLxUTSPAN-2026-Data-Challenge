# Strategy: Using SkillMimic to Predict SPL Landing Outcomes

## Problem

**SPL Competition Data**:
- Has: Body keypoints (hands, joints) throughout shot motion
- Missing: Ball trajectory during shot
- Need: Predict landing outcomes (angle, depth, left/right)

**SkillMimic Data**:
- Has: Body keypoints AND ball trajectory
- Use: Learn relationship between hand kinematics and ball release

## Solution: Transfer Learning Approach

### Step 1: Build Release Prediction Model (SkillMimic)

**Training Data**: Multiple SkillMimic shots

Input features (X):
- Hand position at release
- Hand velocity/speed pre-release (10-20 frames before)
- Peak hand speed timing
- Hand height trajectory
- Arm configuration (elbow/shoulder positions)
- Hand separation (shooting technique)

Target features (y):
- Ball velocity at release: (vx, vy, vz)

**Model**: XGBoost or LightGBM regression
- 3 separate models (one per velocity component)
- Or 1 multi-output model

### Step 2: Detect Release in SPL Data

Since SPL has no contact flag, use hand kinematics:

```python
def detect_release_spl(hand_trajectory):
    """
    Detect release frame from hand motion alone.

    Indicators:
    - Peak hand height (Y coordinate)
    - Peak hand vertical velocity followed by deceleration
    - Wrist extension (hand height > elbow height)
    """
    hand_height = hand_trajectory[:, 1]  # Y coordinate
    hand_vel = calculate_velocity(hand_trajectory)
    hand_vy = hand_vel[:, 1]

    # Find local maxima in height
    peaks = find_peaks(hand_height, prominence=0.1)

    # Filter for peaks with high upward velocity before
    candidates = []
    for peak in peaks:
        if peak > 10 and peak < len(hand_height) - 10:
            # Check velocity was high before peak
            pre_vel = np.max(hand_vy[peak-10:peak])
            if pre_vel > threshold:
                candidates.append(peak)

    return candidates[0] if candidates else np.argmax(hand_height)
```

### Step 3: Extract SPL Features

For each SPL shot:
1. Detect release frame
2. Extract same hand kinematics features as SkillMimic
3. Feed to trained release prediction model
4. Get predicted ball velocity: (vx, vy, vz)

### Step 4: Predict Landing Outcomes

Use ballistic physics to compute trajectory from release to hoop:

```python
def predict_landing_from_release(release_pos, release_vel, hoop_pos):
    """
    Predict where ball lands using physics.

    Args:
        release_pos: (x, y, z) at release
        release_vel: (vx, vy, vz) at release
        hoop_pos: (x_hoop, y_hoop, z_hoop)

    Returns:
        angle: Entry angle at hoop
        depth: Distance from front/back of rim
        left_right: Lateral deviation
    """
    g = 9.81  # Gravity (adjust units if needed)

    # Time to reach hoop height
    # Solve: y(t) = y0 + vy*t - 0.5*g*t^2 = y_hoop
    a = -0.5 * g
    b = release_vel[1]
    c = release_pos[1] - hoop_pos[1]

    t = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)  # Take positive solution

    # Position at hoop
    x_hoop = release_pos[0] + release_vel[0] * t
    z_hoop = release_pos[2] + release_vel[2] * t

    # Velocity at hoop
    vx_hoop = release_vel[0]
    vy_hoop = release_vel[1] - g * t
    vz_hoop = release_vel[2]

    # Landing outcomes
    angle = np.arctan2(vy_hoop, np.sqrt(vx_hoop**2 + vz_hoop**2)) * 180 / np.pi

    # Depth: distance from ideal landing point
    depth = np.sqrt((x_hoop - hoop_pos[0])**2 + (z_hoop - hoop_pos[2])**2)

    # Left/Right: lateral deviation
    left_right = z_hoop - hoop_pos[2]

    return angle, depth, left_right
```

## Implementation Plan

### Phase 1: SkillMimic Analysis (Current)

- [x] Decode SkillMimic observation space
- [x] Find ball position (features 318-320)
- [x] Find ball velocity (features 325-327)
- [x] Extract hand positions from skeleton
- [x] Verify ball-hand distance (0.11-0.14 during contact)
- [x] Create release feature extraction script

### Phase 2: Build Release Dataset

```bash
# If you have multiple SkillMimic .pt files
python build_release_dataset.py \
    --input-dir /path/to/skillmimic/data \
    --output data/skillmimic_release_dataset.csv
```

Creates CSV with:
- Input features: Hand kinematics (30+ features)
- Target features: Ball velocity at release (vx, vy, vz)

### Phase 3: Train Release Predictor

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load dataset
df = pd.read_csv('data/skillmimic_release_dataset.csv')

# Separate features and targets
feature_cols = [col for col in df.columns if not col.startswith('target_')]
target_cols = ['target_ball_vx', 'target_ball_vy', 'target_ball_vz']

X = df[feature_cols]
y = df[target_cols]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train separate model for each velocity component
models = {}
for target in target_cols:
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    model.fit(X_train_scaled, y_train[target])
    models[target] = model

# Save models
import joblib
joblib.dump(models, 'models/release_predictor.pkl')
joblib.dump(scaler, 'models/release_scaler.pkl')
```

### Phase 4: Apply to SPL Data

```python
# Load SPL shot
spl_shot = load_spl_shot(shot_id)  # Your SPL data loader

# Extract hand trajectory
hand_trajectory = extract_hand_keypoints(spl_shot)  # Wrist positions

# Detect release
release_frame = detect_release_spl(hand_trajectory)

# Extract same features as SkillMimic training
spl_features = extract_hand_kinematics_features(
    hand_trajectory,
    release_frame,
    pre_release_window=20
)

# Predict ball velocity at release
X_spl = scaler.transform([spl_features])
predicted_ball_vel = np.array([
    models['target_ball_vx'].predict(X_spl)[0],
    models['target_ball_vy'].predict(X_spl)[0],
    models['target_ball_vz'].predict(X_spl)[0],
])

# Use physics to predict landing
release_pos = hand_trajectory[release_frame]  # Approximate
hoop_pos = [x_hoop, y_hoop, z_hoop]  # Known hoop position

angle, depth, left_right = predict_landing_from_release(
    release_pos,
    predicted_ball_vel,
    hoop_pos
)
```

### Phase 5: Validate and Tune

1. **Check SkillMimic predictions**: Do predicted velocities match actual?
2. **Check SPL predictions**: Do predicted outcomes match actual outcomes?
3. **Tune coordinate systems**: May need to scale/transform SkillMimic → SPL
4. **Feature engineering**: Add more hand kinematics features if needed

## Key Assumptions

### Assumption 1: Coordinate System Alignment

SkillMimic uses heading-relative coordinates (player always faces forward).
SPL might use world coordinates or camera-relative.

**Solution**: Normalize by root position and heading in both datasets.

### Assumption 2: Scale/Units

SkillMimic units might differ from SPL units (meters vs normalized).

**Solution**:
- Learn relative relationships (ratios, angles)
- Normalize velocities by hand speed
- Focus on dimensionless features

### Assumption 3: Release Physics Transfer

Basketball release physics should be similar in simulation vs real world.

**Validation**:
- Check if predicted velocities fall in realistic range (5-7 m/s typical)
- Verify angle predictions are physically plausible (30-50 degrees)

## Expected Performance

**Best case**: SkillMimic physics accurately models real shooting
- Release predictor R² > 0.8
- Landing outcome predictions competitive with direct regression

**Realistic case**: SkillMimic provides useful prior, needs tuning
- Use as initialization for SPL model
- Combine with direct SPL features

**Worst case**: SkillMimic physics don't transfer
- Still useful for feature engineering ideas
- Hand kinematics features still relevant

## Alternative: Hybrid Approach

Don't rely solely on physics prediction. Instead:

**Ensemble**:
1. **Physics-based predictor**: SkillMimic → release → ballistics → outcome
2. **Direct regression**: SPL hand kinematics → outcome (no physics)
3. **Combined**: Weighted average or stacking

This hedges against physics transfer issues while leveraging SkillMimic insights.

## Next Actions

1. **Get more SkillMimic data**: Need multiple shots to train predictor
   - Where is the SkillMimic dataset?
   - How many shots available?

2. **Understand SPL keypoint format**:
   - What keypoints are provided?
   - What coordinate system?
   - What is the sampling rate?

3. **Build and test**:
   - Create release dataset from SkillMimic
   - Train predictor
   - Validate on held-out SkillMimic shots
   - Then apply to SPL

4. **Iterate**:
   - Start simple (just hand height/velocity)
   - Add features incrementally
   - Monitor validation performance
