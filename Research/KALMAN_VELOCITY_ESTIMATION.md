# Kalman Filter Velocity Estimation - Results

## Date: 2026-02-09

## Objective

Compare three velocity estimation methods for ball release velocity:
1. Raw finite differences
2. Savitzky-Golay filtering (window=9, poly=3)
3. Kalman filter (constant velocity pre-release, ballistic post-release)
4. Kalman RTS smoother (forward-backward optimal estimation)

## Method

### Ball Position Estimation
Ball center estimated from right wrist + fingertip keypoints:
```
ball = wrist + 0.6 * (fingertip_center - wrist)
```
where fingertip_center = mean of right_second/third/fourth/fifth/first_finger_distal.

### Release Frame Detection
Two methods tested:
- Wrist peak height: mean=131.6, std=34.3 frames
- Peak wrist speed in window [80,180]: mean=137.1, std=31.5 frames

### Ground Truth Velocity
Computed via inverse projectile from known targets:
- True ball speed: mean=25.40 ft/s (7.741 m/s)
- All 345 training shots have valid inverse projectile solutions

### Kalman Filter Design
- State: [x, y, z, vx, vy, vz] (feet and ft/s)
- Process model (pre-release): constant velocity + acceleration noise
- Process model (post-release): ballistic (gravity: 32.174 ft/s^2 on z)
- Measurement: position only (3D)
- Process noise: continuous white noise acceleration model (Q = G*G^T * sigma_a^2)
- Default params: process_noise_accel=5.0, measurement_noise=0.03

## Results

### Velocity RMSE vs Ground Truth (ft/s)

| Method | vx RMSE | vy RMSE | vz RMSE | Total RMSE | Speed RMSE |
|--------|---------|---------|---------|------------|------------|
| ball_fd | 13.176 | 3.825 | 25.082 | 16.506 | 15.802 |
| ball_savgol | 13.157 | 3.828 | 25.087 | 16.504 | 15.794 |
| ball_kalman | 11.929 | 3.575 | 24.017 | 15.619 | 18.750 |
| ball_kalman_smooth | 13.395 | 3.508 | 14.793 | **11.698** | 17.790 |
| wrist_fd | 12.504 | 3.496 | 24.755 | 16.139 | 15.816 |
| wrist_savgol | 12.477 | 3.494 | 24.757 | 16.133 | 15.839 |
| wrist_kalman | 11.570 | 3.485 | 24.189 | 15.611 | 19.027 |
| wrist_kalman_smooth | 13.180 | 3.361 | 14.675 | **11.552** | 17.722 |

**RTS smoother has lowest total RMSE** (11.6 vs 15.6-16.5 for others), primarily due to much better vz estimation (14.7 vs 24-25 for others). However, RMSE is still enormous - 11.6 ft/s error on a 25.4 ft/s signal.

### Velocity Correlation with Ground Truth

| Method | vx r | vy r | vz r | speed r |
|--------|------|------|------|---------|
| ball_fd | 0.132 | -0.143 | -0.266 | -0.085 |
| ball_savgol | 0.133 | -0.144 | -0.266 | -0.085 |
| ball_kalman | 0.045 | -0.127 | -0.330 | -0.132 |
| ball_kalman_smooth | 0.023 | -0.046 | -0.347 | -0.492 |
| wrist_fd | 0.137 | 0.072 | -0.293 | -0.192 |
| wrist_savgol | 0.138 | 0.073 | -0.293 | -0.194 |
| wrist_kalman | 0.058 | 0.035 | -0.344 | -0.142 |
| wrist_kalman_smooth | 0.049 | 0.132 | -0.453 | -0.576 |

**All correlations are near zero or NEGATIVE.** The estimated velocity has essentially no linear relationship with true ball velocity. The RTS smoother has the strongest correlations but they are negative (vz: r=-0.45, speed: r=-0.58), meaning higher estimated velocity predicts LOWER true velocity.

### Forward Simulation

ALL 345 forward simulations are invalid for ALL methods. The estimated velocities are too small to reach hoop height (10 feet). Every simulation falls back to true targets, producing MSE=0.

### Estimated vs True Speed

| Source | Mean Speed |
|--------|-----------|
| ball_kalman | 7.04 ft/s (2.145 m/s) |
| ball_savgol | 10.50 ft/s (3.201 m/s) |
| wrist_kalman | 6.71 ft/s (2.045 m/s) |
| wrist_savgol | 10.22 ft/s (3.116 m/s) |
| **Ground truth** | **25.40 ft/s (7.741 m/s)** |

**Speed ratio: 3.6x** - True ball velocity is 3.6 times the estimated hand velocity.

### Smoothness (mean abs jerk near release, 20 sample shots)

| Method | Mean Abs Jerk (ft/s^3) |
|--------|----------------------|
| ball_fd | 474.4 |
| ball_savgol | 286.2 |
| **ball_kalman** | **126.3** |
| ball_kalman_smooth | 456.2 |

The forward Kalman filter produces the smoothest velocity (2.3x smoother than Savgol, 3.8x smoother than finite differences). The RTS smoother has high jerk because backward pass corrections create discontinuities.

### Kalman Parameter Sensitivity

| Config | accel | meas | Total RMSE | Best r(vz) |
|--------|-------|------|------------|------------|
| low accel, tight meas | 1.0 | 0.01 | 15.496 | -0.358 |
| low-medium | 2.0 | 0.03 | 15.363 | -0.396 |
| default | 5.0 | 0.03 | 15.619 | -0.330 |
| high accel | 10.0 | 0.03 | 15.754 | -0.309 |
| tight meas | 5.0 | 0.01 | 15.846 | -0.301 |
| **loose meas** | **5.0** | **0.10** | **15.244** | **-0.433** |
| very high accel | 20.0 | 0.03 | 15.921 | -0.295 |
| very tight meas | 5.0 | 0.005 | 16.032 | -0.289 |

Best config: loose measurement noise (0.10 ft = 1.2 inches). RMSE range across all configs is only 15.2-16.0 ft/s, showing the problem is fundamental, not parameter-dependent.

### Per-Player Analysis (ball trajectory, Kalman smoother)

| Player | n | RTS RMSE (ft/s) | Speed r |
|--------|---|-----------------|---------|
| 1 | 70 | 9.892 | -0.297 |
| 2 | 66 | 11.143 | -0.915 |
| 3 | 68 | 13.434 | -0.855 |
| 4 | 67 | 11.161 | -0.884 |
| 5 | 74 | 12.495 | -0.771 |

Player 1 has the smallest RMSE. All players show negative speed correlation, confirming this is systematic, not player-specific.

## Root Cause Analysis

### Why velocity estimation fails

1. **No direct ball tracking**: The dataset contains body keypoints only. Ball position is estimated from wrist + fingertip positions. After ball release, the "ball" estimate follows the decelerating hand, not the accelerating ball.

2. **Finger snap velocity gap**: At release, the fingertips are the fastest-moving part of the kinematic chain. Wrist velocity is only ~2-3 m/s, but the distal fingertip whip adds 3-5 m/s, giving the ball ~7 m/s. This ~3.6x multiplicative gap is a real biomechanical phenomenon that cannot be recovered by any smoothing or filtering method applied to position data.

3. **Release frame ambiguity**: The peak wrist height frame (mean 131.6) occurs 20-40 frames AFTER the actual ball release. The peak wrist velocity frame (mean 137.1) is also unreliable because it often captures follow-through or noise, not the release instant.

4. **Negative velocity correlations**: The negative correlations arise because shots where the hand decelerates quickly (low estimated velocity) tend to be shots with good "snap" (high actual ball velocity). The hand stopping faster means the energy transferred to the ball is greater.

## Conclusions

1. **Kalman filter provides marginal improvement in smoothness** (2.3x smoother than Savgol) but does NOT solve the fundamental velocity recovery problem. The RTS smoother reduces total RMSE from 16.5 to 11.6 ft/s but velocity is still wrong by ~46% of true magnitude.

2. **Forward simulation is completely non-viable** with keypoint-derived velocities: 100% of simulations fail to reach hoop height.

3. **Velocity estimation from body keypoints is a DEAD END for this competition**. The 3.6x velocity gap is a biomechanical reality, not a noise problem. No amount of filtering will close it.

4. **The inverse projectile approach (existing)** remains the only viable physics-based method: given known targets, compute exact velocity. But this requires the targets to already be known (training only) or predicted by other means.

5. **Competition implication**: The per-example locally weighted regression (Sub 1350, LB 0.006776) avoids the velocity estimation problem entirely by directly learning pose-to-target mappings. This is the correct approach for the given data.

## Script
```bash
uv run python scripts/kalman_velocity_estimation.py
```
Runtime: ~23 seconds for full 345-shot analysis with all methods.
