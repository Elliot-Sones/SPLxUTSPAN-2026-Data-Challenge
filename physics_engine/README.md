# Research-Grade Ball Trajectory Simulation Plan

## Goal: Achieve < 0.007 MSE (Current Best: 0.008305)

This plan uses MuJoCo to simulate actual ball trajectories from extracted release parameters, providing direct physics-based predictions for angle, depth, and left_right.

---

## Your Problem Definition

**Input:** 69 body keypoints x 240 frames at 60fps (4 seconds per shot)
**Output:** Predict 3 targets:
- **angle**: Entry angle of ball at hoop (scaled 0-1)
- **depth**: How long/short the ball lands (scaled 0-1)
- **left_right**: Lateral deviation from center (scaled 0-1)

**Training:** 345 shots, 5 players
**Test:** 113 shots

---

## Why Ball Trajectory Simulation Will Work

### Evidence from Your Research

| Finding | Source | Implication |
|---------|--------|-------------|
| DEPTH R² = 0.728 | Player 5, DEFINITIVE_PHYSICS_RESULTS.md | Physics signal exists and is strong |
| Angular momentum: 79.5% variance reduction | angular_momentum_features.py | Dynamics (forces) >> statics (positions) |
| Per-player critical frames differ by 40+ frames | PHYSICS_FEATURES_ANALYSIS.md | Need player-specific extraction |
| Scale factor ~0.29 m/unit | mujoco_basketball_v3.py | Coordinate conversion validated |

### Why Previous Physics Attempts Failed

| Problem | Previous Approach | Research-Grade Fix |
|---------|-------------------|-------------------|
| Fixed release frame | Frame 153 for everyone | Dynamic detection per shot |
| Noisy velocity | 3-frame finite difference | Savitzky-Golay filter (window=7) |
| Wrong scale | Assumed 0.29 m/unit | Multi-segment anthropometric calibration |
| No spin | Ignored | Estimate from fingertip differential |
| Low timestep | 0.002s (500 Hz) | 0.001s (1000 Hz) with RK4 integrator |
| Grid search | Slow, local optima | MJX differentiable physics |

---

## The Core Physics Pipeline

```
SKELETON DATA (69 keypoints x 240 frames)
    |
    v
[PHASE 1: SCALE CALIBRATION]
    - Measure forearm, shin, upper arm in normalized units
    - Compare to known anthropometric lengths
    - Compute player-specific scale_factor (meters/unit)
    |
    v
[PHASE 2: RELEASE DETECTION]
    - Find peak wrist velocity (Savitzky-Golay smoothed)
    - Validate with finger extension rate
    - Player-specific search windows:
        Player 1: frames 110-165
        Player 2: frames 80-195
        Player 3: frames 60-220
        Player 4: frames 125-195
        Player 5: frames 65-200
    |
    v
[PHASE 3: RELEASE PARAMETER EXTRACTION]
    - Position: wrist + fingertip weighted centroid
    - Velocity: Savitzky-Golay derivative at release frame
    - Spin: fingertip velocity differential
    |
    v
[PHASE 4: MUJOCO BALL SIMULATION]
    - Initialize ball at release position
    - Set velocity vector
    - Simulate until ball crosses hoop plane (x = 0)
    - Record landing position (y, z)
    |
    v
[PHASE 5: PHYSICS-TO-TARGET MAPPING]
    - landing_y -> left_right prediction
    - landing_z -> depth prediction
    - entry_angle -> angle prediction
    - Calibrate mapping per player using training data
    |
    v
PREDICTIONS (angle, depth, left_right)
```

---

## Phase 1: Scale Calibration

### The Problem

Your data is normalized (arbitrary units). Real physics needs meters.

### Research-Backed Solution

Use multiple body segments with known anatomical lengths:

| Segment | Keypoints | Typical Length (m) | % of Height |
|---------|-----------|-------------------|-------------|
| Forearm | elbow to wrist | 0.25-0.28 | 14-16% |
| Upper arm | shoulder to elbow | 0.28-0.32 | 16-18% |
| Shin | knee to ankle | 0.38-0.42 | 22-24% |
| Thigh | hip to knee | 0.40-0.46 | 23-26% |

### Implementation

```python
def calibrate_scale_factor(timeseries, keypoint_idx):
    """
    Compute meters-per-unit scale factor using anthropometric references.
    Uses multiple body segments for robustness.
    """
    scales = []

    # Sample stable frames (standing/shooting position)
    for frame in range(100, 180, 10):
        # Forearm length
        elbow = get_joint(timeseries, keypoint_idx, "right_elbow", frame)
        wrist = get_joint(timeseries, keypoint_idx, "right_wrist", frame)
        forearm_len = np.linalg.norm(wrist - elbow)
        if forearm_len > 0.01:
            scales.append(0.265 / forearm_len)  # 26.5cm typical

        # Shin length
        knee = get_joint(timeseries, keypoint_idx, "right_knee", frame)
        ankle = get_joint(timeseries, keypoint_idx, "right_ankle", frame)
        shin_len = np.linalg.norm(ankle - knee)
        if shin_len > 0.01:
            scales.append(0.40 / shin_len)  # 40cm typical

        # Upper arm length
        shoulder = get_joint(timeseries, keypoint_idx, "right_shoulder", frame)
        upper_arm_len = np.linalg.norm(elbow - shoulder)
        if upper_arm_len > 0.01:
            scales.append(0.30 / upper_arm_len)  # 30cm typical

    # Robust estimation: median across all segments and frames
    return np.median(scales) if scales else 0.29  # fallback

def validate_scale(scale_factor, timeseries, keypoint_idx):
    """
    Validate scale factor produces reasonable physical values.
    """
    wrist_height = get_joint(timeseries, keypoint_idx, "right_wrist", 150)[2]
    ankle_height = get_joint(timeseries, keypoint_idx, "right_ankle", 150)[2]

    release_height = (wrist_height - ankle_height) * scale_factor

    # Free throw release height should be 2.0-2.4m
    assert 1.8 < release_height < 2.6, f"Invalid release height: {release_height}m"

    return True
```

---

## Phase 2: Release Frame Detection

### The Problem

Release timing varies by player (frames 110-175 in your data). Fixed frame 153 fails.

### Research-Backed Solution

Multi-criteria detection using:
1. Peak wrist velocity (primary signal)
2. Finger extension rate (secondary confirmation)
3. Player-specific search windows (from your research)

### Implementation

```python
from scipy.signal import savgol_filter

def detect_release_frame(timeseries, keypoint_idx, player_id, scale_factor):
    """
    Detect actual release frame using multiple biomechanical signals.
    """
    # Player-specific search windows (from DEFINITIVE_PHYSICS_RESULTS.md)
    windows = {
        1: (110, 165),
        2: (80, 195),
        3: (60, 220),
        4: (125, 195),
        5: (65, 200),
    }
    start, end = windows.get(player_id, (100, 180))

    # Extract wrist z-positions
    wrist_z = []
    for frame in range(start, end):
        pos = get_joint(timeseries, keypoint_idx, "right_wrist", frame)
        wrist_z.append(pos[2])
    wrist_z = np.array(wrist_z)

    # Smooth and compute velocity using Savitzky-Golay filter
    # window=7 at 60fps = ~0.12s window, polyorder=3 for cubic fit
    wrist_vz = savgol_filter(wrist_z, window_length=7, polyorder=3, deriv=1) * 60

    # Release is at peak upward velocity
    peak_idx = np.argmax(wrist_vz)
    release_frame = start + peak_idx

    # Validate: velocity should decrease after release
    if peak_idx < len(wrist_vz) - 5:
        post_release_vel = np.mean(wrist_vz[peak_idx+1:peak_idx+5])
        if post_release_vel > wrist_vz[peak_idx] * 0.9:
            # Velocity didn't decrease - might be wrong frame
            # Fall back to maximum height
            peak_height_idx = np.argmax(wrist_z)
            release_frame = start + peak_height_idx

    return release_frame, wrist_vz[peak_idx] * scale_factor  # frame, velocity
```

---

## Phase 3: Release Parameter Extraction

### Ball Position at Release

```python
def extract_release_position(timeseries, keypoint_idx, release_frame, scale_factor):
    """
    Extract ball position at release using weighted hand centroid.

    Weighting: 60% palm (wrist + MCP joints), 40% fingertips
    Reference: Your ball_tracking.py implementation
    """
    # Get wrist position
    wrist = get_joint(timeseries, keypoint_idx, "right_wrist", release_frame)

    # Get fingertip positions (index and middle finger distals)
    index_tip = get_joint(timeseries, keypoint_idx, "right_second_finger_distal", release_frame)
    middle_tip = get_joint(timeseries, keypoint_idx, "right_third_finger_distal", release_frame)

    # Weighted centroid (ball is slightly ahead of wrist)
    palm_center = wrist
    fingertip_center = (index_tip + middle_tip) / 2
    ball_pos = 0.6 * palm_center + 0.4 * fingertip_center

    # Get ground reference (ankle position)
    ankle = get_joint(timeseries, keypoint_idx, "right_ankle", release_frame)

    # Convert to meters, with ankle as ground level
    ball_pos_meters = ball_pos * scale_factor
    ball_pos_meters[2] = (ball_pos[2] - ankle[2]) * scale_factor  # Height above ground

    # Free throw line is at x = -4.57m from hoop (hoop at x=0)
    # Adjust x coordinate based on body position
    ball_pos_meters[0] = -4.57  # Assume free throw line

    return ball_pos_meters
```

### Ball Velocity at Release

```python
def extract_release_velocity(timeseries, keypoint_idx, release_frame, scale_factor):
    """
    Extract ball release velocity using Savitzky-Golay differentiation.

    Key insight: Use wider window for more stable estimate, but sample at release frame.
    """
    # Extract wrist positions around release (±10 frames)
    window_start = max(0, release_frame - 10)
    window_end = min(240, release_frame + 10)

    positions = []
    for frame in range(window_start, window_end):
        pos = get_joint(timeseries, keypoint_idx, "right_wrist", frame)
        positions.append(pos)
    positions = np.array(positions)

    # Savitzky-Golay derivative for each axis
    # deriv=1 gives velocity, multiplied by frame rate (60 fps)
    vx = savgol_filter(positions[:, 0], window_length=7, polyorder=3, deriv=1) * 60
    vy = savgol_filter(positions[:, 1], window_length=7, polyorder=3, deriv=1) * 60
    vz = savgol_filter(positions[:, 2], window_length=7, polyorder=3, deriv=1) * 60

    # Get velocity at release frame (center of window)
    idx = release_frame - window_start
    velocity = np.array([vx[idx], vy[idx], vz[idx]]) * scale_factor

    # Coordinate transformation:
    # Our data: X = forward/backward (negative toward hoop)
    # MuJoCo: X = toward hoop (positive)
    velocity_mujoco = np.array([-velocity[0], velocity[1], velocity[2]])

    # Validate: free throw velocity should be 6-8 m/s total
    speed = np.linalg.norm(velocity_mujoco)
    if speed < 4 or speed > 12:
        # Scale to reasonable range
        velocity_mujoco = velocity_mujoco / speed * 7.0

    return velocity_mujoco
```

### Backspin Estimation

```python
def estimate_backspin(timeseries, keypoint_idx, release_frame, scale_factor):
    """
    Estimate ball backspin from fingertip-wrist differential motion.

    Physics: Backspin is created by fingers flicking downward relative to wrist
    """
    # Get fingertip and wrist positions before and after release
    dt = 3  # 3 frames = 0.05 seconds

    index_pre = get_joint(timeseries, keypoint_idx, "right_second_finger_distal", release_frame - dt)
    index_post = get_joint(timeseries, keypoint_idx, "right_second_finger_distal", release_frame + dt)
    wrist_pre = get_joint(timeseries, keypoint_idx, "right_wrist", release_frame - dt)
    wrist_post = get_joint(timeseries, keypoint_idx, "right_wrist", release_frame + dt)

    # Relative fingertip motion (finger velocity - wrist velocity)
    fingertip_motion = (index_post - index_pre) - (wrist_post - wrist_pre)

    # Convert to angular velocity around Y axis (lateral)
    # Backspin is rotation that makes top of ball go backward
    ball_radius = 0.12  # meters

    # Tangential velocity at contact point
    tangential_vel = fingertip_motion[2] * scale_factor * 60 / (2 * dt)

    # Angular velocity (rad/s)
    backspin = tangential_vel / ball_radius

    # Typical backspin: 2-4 Hz = 12-25 rad/s
    backspin = np.clip(backspin, -30, 30)

    return backspin  # rad/s, positive = backspin
```

---

## Phase 4: MuJoCo Ball Simulation

### MuJoCo Scene Configuration

```xml
<mujoco model="basketball_freethrow">
    <!-- Physics settings: 1ms timestep, RK4 for energy conservation -->
    <option gravity="0 0 -9.81" timestep="0.001" integrator="RK4">
        <flag contact="enable"/>
    </option>

    <compiler angle="degree" coordinate="local"/>

    <!-- Default contact parameters -->
    <default>
        <geom condim="4" friction="0.6 0.02 0.01" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </default>

    <worldbody>
        <!-- Ground plane -->
        <geom name="floor" type="plane" size="20 20 0.1" rgba="0.8 0.7 0.6 1"/>

        <!-- Backboard -->
        <body name="backboard" pos="0 0 3.95">
            <geom name="backboard" type="box" size="0.9 0.05 0.6" rgba="1 1 1 0.8"/>
        </body>

        <!-- Rim (modeled as 8 capsule segments for accurate collision) -->
        <body name="rim" pos="0 0.15 3.05">
            <!-- 8 segments forming a circle, radius 0.2286m (9 inches) -->
            <geom name="rim_1" type="capsule" size="0.01" fromto="0.229 0 0 0.162 0.162 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_2" type="capsule" size="0.01" fromto="0.162 0.162 0 0 0.229 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_3" type="capsule" size="0.01" fromto="0 0.229 0 -0.162 0.162 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_4" type="capsule" size="0.01" fromto="-0.162 0.162 0 -0.229 0 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_5" type="capsule" size="0.01" fromto="-0.229 0 0 -0.162 -0.162 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_6" type="capsule" size="0.01" fromto="-0.162 -0.162 0 0 -0.229 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_7" type="capsule" size="0.01" fromto="0 -0.229 0 0.162 -0.162 0" rgba="1 0.3 0 1" friction="0.8"/>
            <geom name="rim_8" type="capsule" size="0.01" fromto="0.162 -0.162 0 0.229 0 0" rgba="1 0.3 0 1" friction="0.8"/>
        </body>

        <!-- Basketball with free joint (6 DOF: translation + rotation) -->
        <body name="ball" pos="-4.57 0 2.0">
            <freejoint name="ball_joint"/>
            <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                  friction="0.6 0.005 0.0001" rgba="1 0.5 0 1"/>
            <inertial pos="0 0 0" mass="0.625" diaginertia="0.003 0.003 0.003"/>
        </body>
    </worldbody>
</mujoco>
```

### Ball Trajectory Simulation

```python
import mujoco

class BasketballSimulator:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Court constants
        self.HOOP_X = 0.0       # Hoop at origin
        self.HOOP_Y = 0.15      # Slightly forward of backboard
        self.HOOP_Z = 3.05      # 10 feet = 3.05 meters
        self.HOOP_RADIUS = 0.2286  # 9 inches

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def simulate_shot(self, release_pos, release_vel, backspin=0):
        """
        Simulate ball trajectory from release to hoop crossing.

        Args:
            release_pos: [x, y, z] in meters (x negative = before hoop)
            release_vel: [vx, vy, vz] in m/s (vx positive = toward hoop)
            backspin: angular velocity in rad/s

        Returns:
            landing_pos: [y, z] at hoop plane crossing (or None if miss)
            entry_angle: angle of ball velocity at entry (degrees)
            trajectory: list of ball positions
        """
        self.reset()

        # Set initial ball position
        self.data.qpos[0:3] = release_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Unit quaternion (no rotation)

        # Set initial velocity
        self.data.qvel[0:3] = release_vel
        self.data.qvel[3:6] = [0, backspin, 0]  # Backspin around Y axis

        trajectory = []
        prev_x = release_pos[0]

        max_time = 3.0  # Maximum 3 seconds of flight

        while self.data.time < max_time:
            mujoco.mj_step(self.model, self.data)

            pos = self.data.qpos[0:3].copy()
            vel = self.data.qvel[0:3].copy()
            trajectory.append(pos)

            # Check if ball crossed hoop plane (x goes from negative to ~0)
            if prev_x < 0.1 and pos[0] >= 0.1:
                # Linear interpolation to exact crossing
                alpha = (0.1 - prev_x) / (pos[0] - prev_x)
                crossing_y = trajectory[-2][1] + alpha * (pos[1] - trajectory[-2][1])
                crossing_z = trajectory[-2][2] + alpha * (pos[2] - trajectory[-2][2])

                landing_pos = np.array([crossing_y, crossing_z])

                # Entry angle: angle of velocity vector from horizontal
                entry_angle = np.degrees(np.arctan2(-vel[2], vel[0]))

                return landing_pos, entry_angle, trajectory

            # Ball hit ground
            if pos[2] < 0.12:
                return None, None, trajectory

            prev_x = pos[0]

        return None, None, trajectory  # Timeout
```

---

## Phase 5: Physics-to-Target Mapping

### Understanding the Target Space

From your training data analysis:
- **angle**: Mean ~45.48°, scaled to 0-1
- **depth**: Mean ~9.66cm from center, scaled to 0-1
- **left_right**: Mean ~-0.78cm, scaled to 0-1

### Mapping Physics Outputs to Targets

```python
def physics_to_targets(landing_pos, entry_angle, player_id, calibration_params):
    """
    Convert physics simulation output to target predictions.

    landing_pos: [y, z] where ball crosses hoop plane
    entry_angle: angle of ball at entry (degrees)
    player_id: 1-5
    calibration_params: per-player learned mapping coefficients
    """
    if landing_pos is None:
        # Simulation failed - return neutral predictions
        return {'angle': 0.5, 'depth': 0.5, 'left_right': 0.5}

    # Get player-specific calibration
    params = calibration_params[player_id]

    # Deviation from hoop center
    y_dev = landing_pos[0] - 0.15  # lateral deviation (meters)
    z_dev = landing_pos[1] - 3.05  # height deviation (meters)

    # ANGLE prediction
    # Entry angle affects outcome - steeper is generally better
    # Optimal entry: 45-52 degrees
    angle_pred = (
        params['angle_intercept'] +
        params['angle_entry_coef'] * entry_angle +
        params['angle_y_coef'] * y_dev +
        params['angle_z_coef'] * z_dev
    )

    # DEPTH prediction
    # Height at entry determines if shot is long or short
    depth_pred = (
        params['depth_intercept'] +
        params['depth_z_coef'] * z_dev +
        params['depth_z2_coef'] * z_dev**2
    )

    # LEFT_RIGHT prediction
    # Lateral deviation directly maps to left_right
    lr_pred = (
        params['lr_intercept'] +
        params['lr_y_coef'] * y_dev
    )

    # Clip to valid range
    return {
        'angle': np.clip(angle_pred, 0, 1),
        'depth': np.clip(depth_pred, 0, 1),
        'left_right': np.clip(lr_pred, 0, 1)
    }
```

### Calibration: Learning the Mapping

```python
from scipy.optimize import differential_evolution
from sklearn.model_selection import GroupKFold

def calibrate_physics_mapping(training_data, simulator):
    """
    Learn per-player mapping from physics outputs to targets.

    Uses differential evolution to find optimal coefficients.
    """
    calibration_params = {}

    for player_id in range(1, 6):
        player_data = [d for d in training_data if d['player_id'] == player_id]

        # Collect physics outputs for this player
        physics_outputs = []
        actual_targets = []

        for shot in player_data:
            # Extract release parameters
            scale = calibrate_scale_factor(shot['timeseries'], shot['keypoint_idx'])
            release_frame, _ = detect_release_frame(
                shot['timeseries'], shot['keypoint_idx'], player_id, scale
            )
            release_pos = extract_release_position(
                shot['timeseries'], shot['keypoint_idx'], release_frame, scale
            )
            release_vel = extract_release_velocity(
                shot['timeseries'], shot['keypoint_idx'], release_frame, scale
            )
            backspin = estimate_backspin(
                shot['timeseries'], shot['keypoint_idx'], release_frame, scale
            )

            # Simulate
            landing, entry_angle, _ = simulator.simulate_shot(
                release_pos, release_vel, backspin
            )

            if landing is not None:
                physics_outputs.append([
                    landing[0], landing[1], entry_angle,
                    landing[0]**2, landing[1]**2  # Quadratic terms
                ])
                actual_targets.append([
                    shot['target_angle'],
                    shot['target_depth'],
                    shot['target_left_right']
                ])

        physics_outputs = np.array(physics_outputs)
        actual_targets = np.array(actual_targets)

        # Fit linear mapping using Ridge regression
        from sklearn.linear_model import Ridge

        # Angle mapping
        angle_model = Ridge(alpha=1.0)
        angle_model.fit(physics_outputs[:, :4], actual_targets[:, 0])

        # Depth mapping
        depth_model = Ridge(alpha=1.0)
        depth_model.fit(physics_outputs[:, [1, 4]], actual_targets[:, 1])

        # Left-right mapping
        lr_model = Ridge(alpha=1.0)
        lr_model.fit(physics_outputs[:, [0]], actual_targets[:, 2])

        calibration_params[player_id] = {
            'angle_intercept': angle_model.intercept_,
            'angle_entry_coef': angle_model.coef_[2],
            'angle_y_coef': angle_model.coef_[0],
            'angle_z_coef': angle_model.coef_[1],
            'depth_intercept': depth_model.intercept_,
            'depth_z_coef': depth_model.coef_[0],
            'depth_z2_coef': depth_model.coef_[1],
            'lr_intercept': lr_model.intercept_,
            'lr_y_coef': lr_model.coef_[0],
        }

    return calibration_params
```

---

## Phase 6: Differentiable Physics (MJX) for End-to-End Optimization

### Why Differentiable Physics?

Instead of manually tuning the skeleton-to-release mapping, learn it end-to-end:

```
skeleton → [learnable extraction] → release params → [physics sim] → landing → [mapping] → prediction
                    ↑                                                                          |
                    |___________________________ gradients ___________________________________|
```

### Implementation with MuJoCo-XLA (MJX)

```python
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
from functools import partial

def setup_differentiable_sim(model):
    """
    Convert MuJoCo model to JAX-compatible differentiable simulation.
    """
    mjx_model = mjx.put_model(model)
    return mjx_model

@jax.jit
def simulate_shot_differentiable(mjx_model, release_params):
    """
    Differentiable forward simulation.

    release_params: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, spin]

    Returns: landing position [y, z]
    """
    # Initialize data
    data = mjx.make_data(mjx_model)

    # Set initial state
    data = data.replace(
        qpos=jnp.array([
            release_params[0], release_params[1], release_params[2],  # position
            1.0, 0.0, 0.0, 0.0  # quaternion
        ]),
        qvel=jnp.array([
            release_params[3], release_params[4], release_params[5],  # velocity
            0.0, release_params[6], 0.0  # angular velocity (backspin)
        ])
    )

    # Simulate 500 steps (0.5 seconds at 1ms timestep)
    def step_fn(data, _):
        data = mjx.step(mjx_model, data)
        return data, data.qpos[0:3]

    final_data, trajectory = jax.lax.scan(step_fn, data, None, length=500)

    # Find hoop crossing (x ~ 0)
    x_positions = trajectory[:, 0]
    crossing_mask = (x_positions[:-1] < 0) & (x_positions[1:] >= 0)
    crossing_idx = jnp.argmax(crossing_mask)

    # Interpolate landing position
    alpha = -x_positions[crossing_idx] / (x_positions[crossing_idx + 1] - x_positions[crossing_idx] + 1e-6)
    landing_y = trajectory[crossing_idx, 1] + alpha * (trajectory[crossing_idx + 1, 1] - trajectory[crossing_idx, 1])
    landing_z = trajectory[crossing_idx, 2] + alpha * (trajectory[crossing_idx + 1, 2] - trajectory[crossing_idx, 2])

    return jnp.array([landing_y, landing_z])

def train_end_to_end(training_data, mjx_model, epochs=100, lr=0.01):
    """
    Train release parameter extraction network end-to-end with physics loss.
    """
    import optax

    # Initialize learnable parameters for extraction
    # These modify how we compute release params from skeleton
    extraction_params = {
        'vel_scale': jnp.array([1.0, 1.0, 1.0]),
        'vel_offset': jnp.array([0.0, 0.0, 0.0]),
        'pos_offset': jnp.array([0.0, 0.0, 0.0]),
        'spin_scale': jnp.array(1.0),
    }

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(extraction_params)

    @jax.jit
    def loss_fn(params, skeleton_features, target_landing):
        # Apply learnable transformation to base release params
        base_vel = skeleton_features['velocity']
        base_pos = skeleton_features['position']
        base_spin = skeleton_features['spin']

        release_params = jnp.concatenate([
            base_pos + params['pos_offset'],
            base_vel * params['vel_scale'] + params['vel_offset'],
            jnp.array([base_spin * params['spin_scale']])
        ])

        # Simulate and compute loss
        predicted_landing = simulate_shot_differentiable(mjx_model, release_params)

        loss = jnp.sum((predicted_landing - target_landing)**2)
        return loss

    grad_fn = jax.grad(loss_fn)

    for epoch in range(epochs):
        total_loss = 0
        for shot in training_data:
            skeleton_features = extract_skeleton_features(shot)
            target_landing = shot['target_landing']  # From calibration

            grads = grad_fn(extraction_params, skeleton_features, target_landing)
            updates, opt_state = optimizer.update(grads, opt_state)
            extraction_params = optax.apply_updates(extraction_params, updates)

            total_loss += loss_fn(extraction_params, skeleton_features, target_landing)

        print(f"Epoch {epoch}: Loss = {total_loss / len(training_data):.6f}")

    return extraction_params
```

---

## Validation Plan

### Step 1: Physics Validation (Before Training)

```python
def validate_physics_accuracy():
    """
    Verify MuJoCo simulation matches analytical projectile motion.
    """
    # Known release parameters
    release_pos = np.array([-4.57, 0, 2.0])  # Free throw line
    release_vel = np.array([6.5, 0, 4.5])    # ~7.7 m/s, 35 degrees

    # Analytical solution (no drag)
    def analytical_trajectory(t):
        x = release_pos[0] + release_vel[0] * t
        y = release_pos[1] + release_vel[1] * t
        z = release_pos[2] + release_vel[2] * t - 0.5 * 9.81 * t**2
        return np.array([x, y, z])

    # MuJoCo simulation
    landing, _, trajectory = simulator.simulate_shot(release_pos, release_vel, backspin=0)

    # Compare at 10 time points
    for i in range(0, len(trajectory), len(trajectory)//10):
        t = i * 0.001  # 1ms timestep
        mujoco_pos = trajectory[i]
        analytical_pos = analytical_trajectory(t)
        error = np.linalg.norm(mujoco_pos - analytical_pos)
        assert error < 0.01, f"Physics error at t={t}s: {error}m"

    print("Physics validation PASSED")
```

### Step 2: Cross-Validation on Training Data

```python
def cross_validate_physics(training_data, simulator):
    """
    GroupKFold CV by player to estimate generalization.
    """
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    player_ids = [d['player_id'] for d in training_data]

    cv_scores = []

    for train_idx, val_idx in gkf.split(training_data, groups=player_ids):
        train_subset = [training_data[i] for i in train_idx]
        val_subset = [training_data[i] for i in val_idx]

        # Calibrate on train
        calibration_params = calibrate_physics_mapping(train_subset, simulator)

        # Evaluate on validation
        mse = 0
        for shot in val_subset:
            # ... extract and simulate ...
            pred = physics_to_targets(landing, entry_angle, shot['player_id'], calibration_params)

            mse += (pred['angle'] - shot['target_angle'])**2
            mse += (pred['depth'] - shot['target_depth'])**2
            mse += (pred['left_right'] - shot['target_left_right'])**2

        cv_scores.append(mse / (3 * len(val_subset)))

    print(f"CV MSE: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")
    return np.mean(cv_scores)
```

### Step 3: Comparison to Baseline

```
Target CV Score: < 0.007 (to beat current 0.008305)

Baseline comparison:
- Current best (ML only): 0.008305
- Physics features only (previous): 0.026 (too high)
- Ball simulation (this approach): Target 0.006-0.007

Success criteria:
1. CV MSE < 0.008 → physics adds value
2. CV MSE < 0.007 → ready to submit
3. CV MSE < 0.006 → significant breakthrough
```

---

## Fallback: Extract Torque Features

If ball simulation doesn't beat baseline, the same pipeline provides torque features:

```python
def extract_torque_features_fallback(timeseries, keypoint_idx, player_id, scale_factor):
    """
    If ball simulation fails, extract torque features for ML instead.

    Uses same IK + inverse dynamics pipeline.
    """
    # Same scale calibration
    # Same release detection
    # But instead of simulating ball...

    # Compute inverse dynamics
    torques, qpos, qvel = compute_inverse_dynamics(
        timeseries, keypoint_idx, model, data, scale_factor
    )

    # Extract features using player-specific windows
    features = extract_torque_features(torques, qvel, player_id)

    # Add to existing ML features
    return features
```

---

## File Structure

```
physics_engine/
    README.md                      # This plan
    __init__.py

    models/
        basketball_court.xml       # MuJoCo scene definition

    core/
        scale_calibration.py       # Phase 1: Normalize to meters
        release_detection.py       # Phase 2: Find release frame
        release_extraction.py      # Phase 3: Position, velocity, spin
        simulator.py               # Phase 4: Ball trajectory simulation
        target_mapping.py          # Phase 5: Physics to predictions

    calibration/
        per_player_calibration.py  # Learn mapping coefficients
        cross_validation.py        # Validate before submission

    mjx/
        differentiable_sim.py      # Phase 6: End-to-end learning
        training.py                # Gradient-based optimization

    scripts/
        run_physics_pipeline.py    # Main execution script
        generate_submission.py     # Create submission file
        validate_physics.py        # Verify simulation accuracy

scripts/
    physics_ball_simulation.py     # Full pipeline script
```

---

## Expected Outcome

| Metric | Previous Physics | This Approach | Target |
|--------|-----------------|---------------|--------|
| CV MSE | 0.026 | 0.006-0.008 | < 0.007 |
| LB Score | 0.007865 (worse than baseline) | < 0.007 | Beat 0.008305 |
| Why better | Ball trajectory not simulated | Direct physics prediction | - |

---

## Key Research References

**Ball Trajectory Physics:**
- ArXiv 1702.07234: "The physics of an optimal basketball free throw"
- NBA optimal: release_angle=58.84°, speed=14.02 fps, backspin=3-5 Hz

**Release Parameter Extraction:**
- Savitzky-Golay filter for velocity: window=5-7, polyorder=3-4
- Multi-segment scale calibration using anthropometric ratios

**MuJoCo Best Practices:**
- Timestep: 0.001s (1000 Hz) with RK4 integrator
- Rim friction: 0.7-0.8, ball friction: 0.6
- 8-segment capsule rim for accurate collision

**Differentiable Physics:**
- MuJoCo-XLA (MJX) for JAX-compatible simulation
- End-to-end gradient optimization of release parameters

**Your Validated Findings:**
- DEPTH R² = 0.728 for Player 5 (physics signal exists)
- Player-specific critical frames (110-220 range)
- Angular momentum transfer: 79.5% variance reduction
